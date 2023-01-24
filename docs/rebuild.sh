rm -f source/mdigest.* source/modules.rst
sphinx-apidoc -o ./source/ ../mdigest/
make html
open build/html/index.html
