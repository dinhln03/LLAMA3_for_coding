from accomodation_website.secrets import DB_PWD

docker_compose = """---
                    version: '3'
                    services:
                      web:
                        build: .
                        publish:
                          - 80
                        links:
                          - db:db
                      db:
                        image: mariadb
                        environment:
                          MYSQL_DATABASE: cannes_db
                          MYSQL_ROOT_PASSWORD: """ + DB_PWD

with open('docker-compose.yml', 'w') as f:
    f.write(docker_compose)
