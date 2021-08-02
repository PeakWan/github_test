"""
WSGI config for Intelligent project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

# os.environ["R_LIBS_USER"]='/www/wwwroot/znfxpt.8dfish.vip/ini/Intelligent/python03_venv/R/lib64/R/library'
os.environ["R_HOME"]='/root/miniconda3/lib/R'
os.environ["PATH"]='/root/miniconda3/lib/R'
# os.environ["LD_LIBRARY_PATH"]='/usr/local/R/lib64/R/lib'
# os.environ["R_LIBS_USER"]='/usr/local/R/lib64/R/library'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Intelligent.settings')
application = get_wsgi_application()
