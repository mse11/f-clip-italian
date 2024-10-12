from streamlit.web import cli as stcli
import os
import sys
# add parent dir, because
#  - configuration_hybrid_clip.py
#  - modeling_hybrid_clip.py
#  removed as duplicity
sys.path.append(os.path.abspath(".."))

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())