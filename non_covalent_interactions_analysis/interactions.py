import numpy as np
from itertools import groupby
from multiprocessing import Pool, cpu_count, current_process
from scipy.spatial import cKDTree
import logging
import concurrent.futures
import pandas as pd
import MDAnalysis as mda
import itertools


def angle_in_range(angle, min_angle, max_angle):
    """Check if an angle value is between min and max angles in degrees"""
    if (min_angle <= angle <= max_angle) or (min_angle <= 180 - angle <= max_angle):
        return True
    return False

def euclidean_distance(atom1, atom2):
            """ compute euclideand distance """
            return np.linalg.norm(atom1 - atom2)


class ParallelHydrogenBondAnalysis:
    def __init__(self, universe, group1, group2, distance=3.6, angle=(0,30), initial_frame=0, final_frame=-1, step=1):
        self.u = universe
        self.u.transfer_to_memory(start=initial_frame, stop=final_frame+1, step=step)

        self.initial_frame = initial_frame
        if final_frame == -1:
            self.final_frame = len(self.u.trajectory)
        else:
            self.final_frame = final_frame
        self.step = step
        
        self.group1_donors = [atom for atom in group1.select_atoms("smarts [O,N,S][H]")]
        self.group2_donors = [atom for atom in group2.select_atoms("smarts [O,N,S][H]")]
        self.group1_acceptors = [atom for atom in group1.select_atoms("smarts [O,N,*-;!+]")]
        self.group2_acceptors = [atom for atom in group2.select_atoms("smarts [O,N,*-;!+]")]

        self.distance = distance
        self.angle = angle
        self.results = []

    def _write_log(self, message):
        with open("output.log", "a") as f:
            f.write(message + "\n")        

    def _compute_angle(self, donor, h, acceptor):
        v1 = h.position - donor.position
        v2 = h.position - acceptor.position
        dot_product = np.dot(v1, v2)

        # Check if the dot product is within the valid range
        if -1 <= dot_product <= 1:
            angle_rad = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            return np.degrees(angle_rad)
        else:
            # Handle cases where the dot product is slightly out of range
            if dot_product < -1:
                return 180.0  # The angle is 180 degrees
            else:
                return 0.0    # The angle is 0 degrees

    def _analyze_frame(self, frame_number):
        
        frame_data = []
        ts = self.u.trajectory[frame_number]
        self._write_log(f"Analyzing frame:{ts.frame} {ts}\n")
        processed_pairs = set()

        try:    
            for donor_group, acceptor_group in [(self.group1_donors, self.group2_acceptors), (self.group2_donors, self.group1_acceptors)]:
                # Check if this pair has already been processed in reverse order
                pair_key = (tuple(donor_group), tuple(acceptor_group))
                reverse_pair_key = (tuple(acceptor_group), tuple(donor_group))
                
                if pair_key in processed_pairs or reverse_pair_key in processed_pairs:
                    continue  # Skip this pair as it has already been processed
                    
                # Add the current pair to the set of processed pairs
                processed_pairs.add(pair_key)

                donor_hydrogen_pairs = [(donor, h) for donor in donor_group for h in donor.bonded_atoms if h.name.startswith('H')]
                acceptor_tree = cKDTree([acceptor.position for acceptor in acceptor_group])

                for donor, h in donor_hydrogen_pairs:
                    acceptor_indices = acceptor_tree.query_ball_point(h.position, self.distance)

                    for index in acceptor_indices:
                        acceptor = acceptor_group[index]
                        
                        bond_angle = self._compute_angle(donor, h, acceptor)
                        
                        if angle_in_range(bond_angle, self.angle[0], self.angle[1]):
                        
                            frame_data.append({
                                'donor': f'{donor.chainID}:{donor.resname}{donor.resid}-{donor.name}--{donor.index}',
                                'h': f'{h.chainID}:{h.resname}{h.resid}-{h.name}--{h.index}',
                                'acceptor': f'{acceptor.chainID}:{acceptor.resname}{acceptor.resid}-{acceptor.name}--{acceptor.index}',
                                'frame': frame_number,
                                'interaction': 'hydrogen_bond'
                            })

        except Exception as e:
            print(f"Error in frame {ts}: {e}")

            error_log = f"Error in frame {frame_number}: {e}"
            logging.error(error_log)

        return frame_data

    def run(self):
        frames = []
        for ts in self.u.trajectory[self.initial_frame: self.final_frame+1: self.step]:
            frames.append(ts.frame)

        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            results = list(executor.map(self._analyze_frame, frames))

        # Flatten the list of lists
        self.results = [item for sublist in results for item in sublist]

    def count_by_time(self):
        if not self.results == []:
            df = pd.DataFrame(self.results)

            # Calculate the total number of frames
            total_frames = len(self.u.trajectory[self.initial_frame:self.final_frame+1:self.step])

            # Group by hydrogen bond and count the number of frames for each
            counts = df.groupby(['donor', 'h', 'acceptor']).size().reset_index(name='count')

            # Calculate the percentage for each hydrogen bond
            counts['percentage'] = (counts['count'] / total_frames) * 100

            return counts
        
        else:
            return None
    


class ParallelPiPiStackingAnalysis:
    def __init__(self, universe, group1, group2, distance=4.2, f2f_angle=(0, 45), e2f_angle=(45, 135), initial_frame=0, final_frame=-1, step=1):
        self.u = universe
        self.u.transfer_to_memory(start=initial_frame, stop=final_frame + 1, step=step)

        self.initial_frame = initial_frame
        if final_frame == -1:
            self.final_frame = len(self.u.trajectory)
        else:
            self.final_frame = final_frame
        self.step = step

        # Select atoms based on SMARTS patterns for rings
        self.rings = list(zip(self.u.select_atoms("(smarts [a]1:[a]:[a]:[a]:[a]:[a]:1) or (smarts [a]1:[a]:[a]:[a]:[a]:[a]:1) or (smarts [a]1:[a]:[a]:[a]:[a]:[a]:[a]:1)\
                                                  or smarts [c,n]1[c,n][c,n][c,n][c,n][c,n]1").resids,
                    self.u.select_atoms("(smarts [a]1:[a]:[a]:[a]:[a]:[a]:1) or (smarts [a]1:[a]:[a]:[a]:[a]:[a]:1) or (smarts [a]1:[a]:[a]:[a]:[a]:[a]:[a]:1)\
                                        or smarts [c,n]1[c,n][c,n][c,n][c,n][c,n]1").atoms.names))

        residue_groups = {}
        for resid, name in self.rings:
            if resid not in residue_groups:
                residue_groups[resid] = []
            residue_groups[resid].append(name)

        # Generate atom groups for rings in each residue
        self.residue_atom_groups = {}
        self.residue_cm = {}
        for resid, names in residue_groups.items():
            group = self.u.select_atoms(f"resid {resid} and name {' '.join(names)}") 
            self.residue_atom_groups.update({resid:  group})
            self.residue_cm.update({resid: group.center_of_mass()})

        # Generate the final pair list by constructing pairs of keys and corresponding pairs of residues
        self.ring_pairs = {}
        for pair_keys in itertools.combinations(residue_groups.keys(), 2):
            keys = list(pair_keys)
            if euclidean_distance(self.residue_cm[keys[0]], self.residue_cm[keys[1]]) < 10.0:
                pair_values = [self.residue_atom_groups[keys[0]], self.residue_atom_groups[keys[1]]]
                self.ring_pairs.update({tuple(keys): pair_values})

        self.distance = distance
        self.f2f_angle = f2f_angle
        self.e2f_angle = e2f_angle
        self.results = []

    def _write_log(self, message):
        with open("output.log", "a") as f:
            f.write(message + "\n")   

    def _compute_angle(self, centroid1, centroid2):
        v1 = centroid2 - centroid1
        v2 = [0.0, 0.0, 1.0]  # Define the z-axis (e.g., perpendicular to the aromatic plane)

        angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return np.degrees(angle_rad)

    def _analyze_frame(self, frame_number):
        
        frame_data = []
        ts = self.u.trajectory[frame_number]
        self._write_log(f"Analyzing frame:{ts.frame} {ts}\n")
        
        try:
            for _, ring_pair in self.ring_pairs.items():
        
                # Calculate the centroid of the aromatic rings
                centroid1 = np.mean(list(ring_pair)[0].positions, axis=0)
                centroid2 = np.mean(list(ring_pair)[1].positions, axis=0)
                
                # Calculate the distance between centroids
                distance = np.linalg.norm(centroid1 - centroid2)

                # Calculate the angle between the aromatic planes
                angle = self._compute_angle(centroid1, centroid2)

                if (distance <= self.distance and angle_in_range(angle, self.f2f_angle[0], self.f2f_angle[1])) or \
                    (distance <= self.distance and angle_in_range(angle, self.e2f_angle[0], self.e2f_angle[1])):

                    donor    = ring_pair[0]
                    acceptor = ring_pair[1]
                    frame_data.append({
                        'donor'      : f'{np.unique(donor.chainIDs)[0]}:{np.unique(donor.resnames)[0]}{np.unique(donor.resids)[0]}',
                        'acceptor'   : f'{np.unique(acceptor.chainIDs)[0]}:{np.unique(acceptor.resnames)[0]}{np.unique(acceptor.resids)[0]}',
                        'frame'      : frame_number,
                        'interaction': 'pi_stacking'
                    })

        except Exception as e:
            print(f"Error in frame {ts}: {e}")

        return frame_data

    def run(self):
        frames = []
        for ts in self.u.trajectory[self.initial_frame: self.final_frame+1: self.step]:
            frames.append(ts.frame)

        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            results = list(executor.map(self._analyze_frame, frames))

        # Flatten the list of lists
        self.results = [item for sublist in results for item in sublist]

    def count_by_time(self):
        if self.results == []:
            return None
        else:
            df = pd.DataFrame(self.results)

            # Calculate the total number of frames
            total_frames = len(self.u.trajectory[self.initial_frame:self.final_frame + 1:self.step])

            # Group by pi-pi stacking interaction and count the number of frames for each
            counts = df.groupby(['donor', 'acceptor']).size().reset_index(name='count')

            # Calculate the percentage for each pi-pi stacking interaction
            counts['percentage'] = (counts['count'] / total_frames) * 100

            return counts
    

class ParallelHydrophobicInteractionAnalysis:
    def __init__(self, universe, group1, group2, distance=3.6, initial_frame=0, final_frame=-1, step=1):
        self.u = universe
        self.u.transfer_to_memory(start=initial_frame, stop=final_frame+1, step=step)

        self.initial_frame = initial_frame
        if final_frame == -1:
            self.final_frame = len(self.u.trajectory)
        else:
            self.final_frame = final_frame
        self.step = step
        
        self.distance = distance
        self.cm_pairs = {}
        self.results = []

        atom_selection = "smarts [C,S,F,Cl,Br,I]"

        def calculate_center_of_mass(group, hydrophobic_atom_selection):
            
            # Calculate center of mass for each residue-atoms in the group
            name_res = list(zip(group.select_atoms(hydrophobic_atom_selection).atoms.names, group.select_atoms(hydrophobic_atom_selection).atoms.resids))
            group_resid_dict = dict(zip(np.unique(group.select_atoms(hydrophobic_atom_selection).atoms.resids), [list(zip(*g))[0] for k, g in groupby(name_res, lambda x: x[1])]))
            
            group_cm = {}
            for res, atoms in group_resid_dict.items():
                ag = group.select_atoms('resid ' + str(res) + ' and name ' + ' '.join(x for x in atoms))
                
                cm = ag.center_of_mass()
                group_cm.update({res: (ag, cm)})

            return group_cm

        group1_cm = calculate_center_of_mass(group1, atom_selection)
        group2_cm = calculate_center_of_mass(group2, atom_selection)  
        
        # generate pairs of values of group1_cm and group2_cm
        
        for res1, cm1 in group1_cm.items():
            for res2, cm2 in group2_cm.items():
                if res1 != res2:
                    
                    residue_pair = sorted(np.asarray([res1, res2]))

                    if euclidean_distance(cm1[1], cm2[1]) < 10.0:
                        self.cm_pairs.update({tuple(residue_pair) : tuple([cm1[0], cm2[0]])})
        


    def _write_log(self, message):
        with open("output.log", "a") as f:
            f.write(message + "\n")           

    def _analyze_frame(self, frame_number):
        
        frame_data = []
        ts = self.u.trajectory[frame_number]
        self._write_log(f"Analyzing frame:{ts.frame} {ts}\n")
        
        try:
            for (res_d, res_a), (ag_d, ag_a), in self.cm_pairs.items():
                
                # print('ag_d', ag_a, flush=True)
                # print('ag_a', ag_d, flush=True)
                # print('com_d', com_d, flush=True)
                # print('com_a', com_a, flush=True)                
                
                donor_com = ag_d.center_of_mass() 
                acceptor_com = ag_a.center_of_mass()
                
                distance = np.linalg.norm(donor_com - acceptor_com)
                
                donor = ag_d 
                acceptor = ag_a 
                
                if distance <= self.distance:
                    
                    frame_data.append({
                                'donor': f'{np.unique(donor.chainIDs)[0]}:{np.unique(donor.resnames)[0]}{np.unique(donor.resids)[0]}',
                                'acceptor': f'{np.unique(acceptor.chainIDs)[0]}:{np.unique(acceptor.resnames)[0]}{np.unique(acceptor.resids)[0]}',
                                'frame': frame_number,
                                'interaction': 'hydrophobic'
                            })

        except Exception as e:
            print(f"Error in frame {ts}: {e}")

        return frame_data

    def run(self):
        frames = []
        for ts in self.u.trajectory[self.initial_frame: self.final_frame+1: self.step]:
            frames.append(ts.frame)
            

        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            results = list(executor.map(self._analyze_frame, frames))

        # Flatten the list of lists
        self.results = [item for sublist in results for item in sublist]

    def count_by_time(self):
        if self.results == []:
            return None
        else:
            df = pd.DataFrame(self.results)

            # Calculate the total number of frames
            total_frames = len(self.u.trajectory[self.initial_frame:self.final_frame+1:self.step])

            # Group by interaction and count the number of frames for each
            counts = df.groupby(['donor', 'acceptor']).size().reset_index(name='count')

            # Calculate the percentage for each interaction
            counts['percentage'] = (counts['count'] / total_frames) * 100

            return counts