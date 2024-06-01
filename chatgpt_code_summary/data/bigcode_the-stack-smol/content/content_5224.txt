import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from src.DbHelper import DbHelper

persons = [
    'Lucy',
    'Franz',
    'Susanne',
    'Jonathan',
    'Max',
    'Stephan',
    'Julian',
    'Frederike',
    'Amy',
    'Miriam',
    'Jonas',
    'Anna',
    'Sebastian'
]

addresses = [ f'Musterstraße {i}' for i in range(1,11)]
accounts = [ f'Bank Account {i}' for i in range(1, 14)]
phones = [f'Phone Number {i}' for i in range(1,12)]
creditcards = [f'Credit Card Number {i}' for i in range(1,14)]
socialsecuritynumbers = [f'SSN {i}' for i in range(1,10)]


nodes = {
    'Person':('name', persons),
    'Address':('address', addresses),
    'BankAccount':('account', accounts),
    'CreditCard':('number', creditcards),
    'SSN':('ssn', socialsecuritynumbers)
}


if __name__ == "__main__":
    # See https://neo4j.com/developer/aura-connect-driver/ for Aura specific connection URL.
    scheme = "neo4j"  # Connecting to Aura, use the "neo4j+s" URI scheme
    host_name = "localhost"
    port = 7687  # Bolt Port https://neo4j.com/docs/operations-manual/current/configuration/ports/ | .NET | Java | JavaScript | Go | Python

    url = f"{scheme}://{host_name}:{port}"
    user = 'neo4j'
    password = 'neo4j'
    db_helper = DbHelper(url, user, password)

    for Label, values in nodes.items():
        PropertyKey = values[0]
        for PropertyValue in values[1]:
            db_helper.run_query(
                'CREATE (node:' + Label + ' {' + PropertyKey + ': "' + PropertyValue + '" }) RETURN node.' + PropertyKey
            )

    db_helper.close()
