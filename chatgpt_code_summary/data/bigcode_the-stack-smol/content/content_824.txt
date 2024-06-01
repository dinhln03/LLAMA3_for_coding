# -*- coding: utf-8 -*-
import os
import sys
import datetime

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from django.conf import settings
from app1.models import Thing


class Command(BaseCommand):
    args = '<id name>'
    help = 'create or update thing model.'
    use_settings = 'settings'

    def handle(self, *args, **options):
        """
        finished when raise CommandError, exit code = 1.
        other exit code = 0
        """
        _retcode = 1
        _dbname = 'default'
        
        try:
            print('settings.ENV_MODE = %s' % (settings.ENV_MODE))
            print('settings.DATABASES = %s' % (settings.DATABASES))
            
            _id = int(args[0])
            _name = args[1]

            print('id: %s, name:%s' % (_id, _name))
            
            qs = Thing.objects.filter(id=_id)
            _nowdt = timezone.now()
            if 0 < len(qs):
                print('do update.')
                _r = qs[0]
                
                # _r.id
                _r.name = _name
                # _r.create_at
                _r.update_at = _nowdt
                _r.save(using=_dbname)
            else:
                print('do insert.')
                
                if _id < 1:
                    _id = None
                    
                _t = Thing(
                    id=_id,
                    name=_name,
                    create_at=_nowdt,
                    update_at=_nowdt)
                _t.save(using=_dbname)
        except:
            print('EXCEPT: %s(%s)' % (sys.exc_info()[0], sys.exc_info()[1]))
            print('finished(ng)')
            raise CommandError('ng')

        # raise CommandError('ok')
        print('finished(ok)')
        sys.exit(0)
