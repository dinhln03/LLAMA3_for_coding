#!/usr/bin/env python

import common
import json
import docker_utils

nginx_sites_available = '/etc/nginx/sites-available'
CERT_DIR = '/root/certs'

import subprocess

def create_certificates(domains):
    format_args = {'cert_dir': CERT_DIR}

    import os.path
    if not os.path.isfile(os.path.join(CERT_DIR, 'acmeCA.key.deleteme')):
        commands = """openssl rsa -in %(cert_dir)s/acmeCA.key -out %(cert_dir)s/acmeCA.key.deleteme""" % format_args
        for command in [cmd for cmd in commands.split("\n") if cmd]:
            subprocess.call([arg for arg in command.split(" ") if arg])

    for domain in domains:
        create_certificate(domain)

def create_certificate(domain):
    format_args = {'domain': domain,
                   'cert_dir': CERT_DIR}
    import os.path
    if os.path.isfile('%(cert_dir)s/%(domain)s.key' % format_args):
        return
    
    commands = """
    openssl genrsa -out %(cert_dir)s/%(domain)s.key 2048
    openssl req -new -key %(cert_dir)s/%(domain)s.key -out %(cert_dir)s/%(domain)s.csr  -subj /C=DE/ST=Niedersachsen/L=Osnabrueck/O=OPS/CN=%(domain)s
    openssl x509 -req -in %(cert_dir)s/%(domain)s.csr -CA %(cert_dir)s/acmeCA.pem -CAkey %(cert_dir)s/acmeCA.key.deleteme -CAcreateserial -out %(cert_dir)s/%(domain)s.crt -days 500
    rm %(cert_dir)s/%(domain)s.csr
""" % format_args
    
    for command in [cmd for cmd in commands.split("\n") if cmd]:
        print command.split(" ")
        subprocess.call([arg for arg in command.split(" ") if arg])

# create_certificates([host.domains[0] for host in common.get_vhost_config()])

def update_vhosts_config(applications):
    jsonFile = open('/root/config/nginx_vhosts.json', "r")
    data = json.load(jsonFile)
    jsonFile.close()

    for app in applications:
        docker_container_config = docker_utils.get_config(app.docker_container_name)
        vhost_config = data[app.vhost_name]
        vhost_config['port'] = docker_container_config.port if not app.docker_container_port else app.docker_container_port
        vhost_config['ip_addr'] = docker_container_config.ip_addr
        
    jsonFile = open('/root/config/nginx_vhosts.json', "w+")
    jsonFile.write(json.dumps(data, indent=4, sort_keys=True))
    jsonFile.close()


def update_vhosts(vhosts):
    for vhost in vhosts:
        host = vhost.host
        port = vhost.port
        ip_addr = vhost.ip_addr
        domains = vhost.domains
        flags = vhost.flags

        location_tmpl = """
          location    %(path)s {
            proxy_pass  http://upstream_%(upstream)s%(upstream_path)s;
            proxy_http_version 1.1;
            %(redirect_rule)s
            proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
            proxy_set_header        Host            %(host)s;
            %(set_script_name)s
            proxy_set_header        X-Real-IP       $remote_addr;
            proxy_set_header        X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Port $server_port;
            %(misc)s
          }
    """
        location_tmpl_params = {
            'redirect_rule': 'proxy_redirect   off;' if flags.get('disableRedirect') else ''
            }

        def render_location(location_dict):
            location_dict['host'] = location_dict.get('host', '$host')
            location_dict['set_script_name'] = location_dict.get('set_script_name', '')
            location_dict['misc'] = location_dict.get('misc', '')
            location_dict['upstream_path'] = location_dict.get('upstream_path', '')
            params = dict(location_dict.items()+ location_tmpl_params.items())
            # print params
            return location_tmpl % params
    
        location_parameters = { 'upstream': domains[0], 'path': '/', 'host': flags.get('forceHost', '$host'),
                                'upstream_path': flags.get('upstream_path', '')}

        if 'htpasswd_file' in flags:
            location_parameters['misc'] = 'auth_basic "Restricted"; auth_basic_user_file %s;' % (flags['htpasswd_file'])

        if 'location_extra' in flags:
            location_parameters['misc'] = location_parameters['misc'] if 'misc' in location_parameters else ''
            location_parameters['misc'] += flags['location_extra']

        location = render_location(location_parameters)
    
        location_ssl = location

        upstreams = [{
                'local_port': port,
                'local_address': ip_addr,
                'name': domains[0]
                }]

        if flags.get('sslToPort'):
            upstream_name = "%s_ssl " % domains[0]
            location_ssl = render_location({ 'upstream': upstream_name, 'path': '/', 'host': flags.get('forceHost', '$host')})
            upstreams.append({
                    'local_port': flags.get('sslToPort'),
                    'local_address': ip_addr,
                    'name': upstream_name
                    })

        if flags.get('httpsToHttpPaths'):
            for path in flags.get('httpsToHttpPaths').split(','): 
               location_ssl += "\n" + render_location({ 'upstream': domains[0], 'path': '/%s' % path, 'host': flags.get('forceHost', '$host') })

        other_locations = [{ 'upstream': domains[0], 'path': '@failover', 'host': flags.get('forceHost', '$host')}]
        other_locations_https = []

        path_idx = 0
        for path, path_config in vhost.paths.items():
            upstream_name = "%s_%s " % (domains[0], path_idx)
            upstreams.append({
                    'local_port': path_config['port'],
                    'local_address': vm_map[path_config['host']]['local_address'],
                    'name': upstream_name
                    })

            if path_config['secure']:
                other_locations_https.append({ 'upstream': upstream_name, 'path': '/%s' % path,
                                               'misc': '''
''',
                                     'set_script_name': ('proxy_set_header        SCRIPT_NAME     /%s;' % path.rstrip('/')) if path_config.get('setScriptName') else '',
                                     'host': flags.get('forceHost', '$host')})
            else:
                other_locations.append({ 'upstream': upstream_name, 'path': '/%s' % path,
                                         'misc': '''
	    error_page 500 = @failover;
	    proxy_intercept_errors on;
''',
                                     'set_script_name': ('proxy_set_header        SCRIPT_NAME     /%s;' % path.rstrip('/')) if path_config.get('setScriptName') else '',
                                     'host': flags.get('forceHost', '$host')})
                

            path_idx += 1

        upstream_tmpl = 'upstream upstream_%(name)s { server %(local_address)s:%(local_port)s; }'

        rewrites = ''

        extra_directives = ''
        if flags.get('block_robots'):
            extra_directives += '''
            location = /robots.txt {
                alias /var/www/robots_deny.txt;
            }
            '''

        if flags.get('allow_robots'):
            extra_directives += '''
            location = /robots.txt {
                alias /var/www/robots_allow.txt;
            }
            '''

        if 'server_config_extra' in flags:
            extra_directives += flags['server_config_extra']

        if flags.get('aliases'):
            aliases = flags.get('aliases').split("\n")
            for alias in aliases:
                extra_directives += '''
            location /%s {
                alias %s;
            }
            ''' % tuple(alias.strip().split('->'))

    
        if vhost.rewrites:
            rewrites += vhost.rewrites

        location_http = location if flags.get('allow_http') else 'return 301 https://$host$request_uri;'

        if flags.get('httpPaths'):
            for path in flags.get('httpPaths').split(','): 
                location_http = "\n" + render_location({ 'upstream': domains[0], 'path': '/%s' % path, 'host': flags.get('forceHost', '$host') }) + "\n" + '''                                                                                                                              location  / { return 301 https://$host$request_uri; }    
            '''

        format_args = {
            'upstreams': "\n".join([upstream_tmpl % up for up in upstreams]),
            'public_port': port,
            'other_locations': "\n".join([render_location(location_dict) for location_dict in other_locations]),
            'other_locations_https': "\n".join([render_location(location_dict) for location_dict in other_locations_https]),
            'extra_directives': extra_directives,
            'domain': domains[0],
            'server_names': ' '.join(domains) if not flags.get('rewriteDomains') else domains[0],
            'location': location_ssl,
            'rewrites': rewrites,
            'upload_limit': flags.get('uploadLimit', '20M'),
            'location_http': location_http,
            'cert_dir': CERT_DIR}
    
    
        config = """
        %(upstreams)s
        server {
          listen      80;
          server_name %(server_names)s;
          client_max_body_size %(upload_limit)s;

          %(rewrites)s

          %(location_http)s

          %(other_locations)s

          %(extra_directives)s
        }
        
    """ % format_args

        if not flags.get('noSsl'):
            config += """
        server {
          listen      443 ssl;
          server_name %(server_names)s;
          client_max_body_size %(upload_limit)s;
        
          ssl on;
          ssl_certificate     %(cert_dir)s/%(domain)s.cer;
          ssl_certificate_key %(cert_dir)s/%(domain)s.key;
          ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-RC4-SHA:ECDHE-RSA-AES128-SHA:AES128-GCM-SHA256:RC4:HIGH:!MD5:!aNULL:!EDH:!CAMELLIA;
          ssl_protocols TLSv1.2 TLSv1.1 TLSv1;
          ssl_prefer_server_ciphers on;
    
          %(location)s

          %(other_locations_https)s

          %(extra_directives)s
        }
    """ % format_args


        if flags.get('rewriteDomains'):
            for domain in domains[1:]:
                config += """
server {
        listen 80;
        server_name %(domain1)s;
        return 301 http://%(domain2)s$request_uri;
}
""" % {'domain1': domain, 'domain2': domains[0]}



        f = open('%s/%s' % (nginx_sites_available, domains[0]), 'w')
        f.write(config)
        f.close()
    
    '''
            proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
            proxy_set_header        Host            $host;
            proxy_set_header        X-Real-IP       $remote_addr;
            proxy_set_header        X-Forwarded-For $proxy_add_x_forwarded_for;
    '''

update_vhosts_config(common.get_applications_config())
update_vhosts(common.get_vhost_config())
