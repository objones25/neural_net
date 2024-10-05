import sys
import os

# Add the site-packages directory to sys.path
site_packages_dir = '/opt/anaconda3/envs/py312/lib/python3.12/site-packages'
if site_packages_dir not in sys.path:
    sys.path.insert(0, site_packages_dir)