import user
import db

if __name__ == "__main__":
  # Initializes the database if it doesn't already exist
  engine = db.open_db('maintenance.db')
  db.create_tables(engine)

  # TODO: Make this selectable with arrow keys
  while True:
    print('\nSelect an option:\n1. View Service History\n2. Add a Car\n3. Add a Service\n4. Exit')
    userInput = input()
    
    if userInput[0] == '1': user.view_services(engine)
    elif userInput[0] == '2': user.insert_car(engine)
    elif userInput[0] == '3': user.insert_service(engine)
    elif userInput[0] == '4': break