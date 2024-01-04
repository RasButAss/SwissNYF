import tempfile
import subprocess
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
subprocess.call(['cp', f'{currentdir}/.env.template', f'{currentdir}/.env'])
t = f'{currentdir}/.env'
try:
    editor = os.environ['EDITOR']
except KeyError:
    editor = 'nano'
subprocess.call([editor, t])
