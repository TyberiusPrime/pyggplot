import pandoc
import os

pandoc.core.PANDOC_PATH = '/path/to/pandoc'

doc = pandoc.Document()
doc.markdown = open('README.md').read()
f = open('README.txt','w+')
f.write(doc.rst)
f.close()
os.system("setup.py sdist upload")
os.remove('README.txt')
