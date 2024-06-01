
# coding: utf-8

# In[ ]:


def choice():
    print("1-create,2-update,3-read,4-delete")
    try:
       x=int(input("\nEnter your choice:"))
    except ValueError:
        print("Enter integer choice:....")
        choice()
    else:
        if(x==1):
            create()
        elif(x==2):
            update()
        elif(x==3):
            read()
        else:
            delete()
    
def create():
    try:
        id=int(input("\nEnter your id:"))
    except ValueError:
        f=int(input("Enter a valid integer number or press 0 to exit:"))
        if(f==0):
            choice()
        else:
            create()
    else:
        name=str(input("Enter your name:"))
        college=str(input("Enter the college name:"))
        branch=str(input("Enter the branch:"))
        print("\n")
        lid.append(id)
        lname.append(name)
        lcollege.append(college)
        lbranch.append(branch)
    choice()
def update():
    try:
        id=int(input("Enter your id:"))
    except ValueError:
        print("\nEnter valid integer  id.......")
        update()
    else:
        if id in lid:
            r=lid.index(id)
            newname=str(input("Enter the name"))
            lname[r]=newname
            newcollege=str(input("Enter the college name:"))
            lcollege[r]=newcollege
            newbranch=str(input("Enter the branch:"))
            lbranch[r]=newbranch
        else:
            print("id didnot match........")
            print("please register yourself....")
    choice()
    
def read():
    try:
        db=int(input("\nTo access database enter id:"))
    except ValueError:
        print("Enter integer id.....")
        read()
    else:   
        if db in lid:
            print("ID:-",lid)
            print("NAMES:-",lname)
            print("COLLEGE:-",lcollege)
            print("BRANCH:-",lbranch)
        elif(lid==dummy):
            print("\nno records......")
        else:
            print("\nRegister inorder to access database.....")
    choice()
def delete():
    if(lid==dummy):
        print("No records found to delete.....")
    else:
        try:
            id=int(input("\nEnter your id:"))
        except ValueError:
            print("\nEnter the valid integer id.....")
            delete()
        else:
            if id in lid:
                delid.append(id)
                d=lid.index(id)
                del lid[d]
                del lname[d]
                del lcollege[d]
                del lbranch[d]
                print("\ndetails of your id has been deleted sucessfully......")
            elif id in delid:
                print("\nDetails of this id has been deleted......")
            else:
                print("\nregister the id... ")
    choice()
#creating lists    
lid=[]
lname=[]
lcollege=[]
lbranch=[]
dummy=[]   
delid=[]  #list of deleted id
choice()

