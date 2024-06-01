import mysql_conn

class BaseField:
    def __init__(self,name,column_type,primary_key,default):
        self.name=name
        self.column_type=column_type
        self.primary_key=primary_key
        self.default=default

class StringField(BaseField):
    def __init__(self,name,column_type='varchar(200)',primary_key=False,default=None):
        super().__init__(name,column_type,primary_key,default)

class IntegerField(BaseField):
    def __init__(self,name,column_type='int',primary_key=False,default=0):
        super().__init__(name, column_type, primary_key, default)


class ModelsMeta(type):
    def __new__(cls,name,bases,attr):
        if name=='Models':
            return type.__new__(cls,name,bases,attr)

        table_name=attr.get('table_name',None)
        if not table_name:
            table_name=name

        primary_key=None
        mappings=dict()
        for k,v in attr.items():
            if isinstance(v,BaseField):
                mappings[k]=v
                if v.primary_key:
                    if primary_key:
                        raise TypeError('主键重复')
                    primary_key=k

        for k in mappings.keys():
            attr.pop(k)

        if not primary_key:
            raise TypeError('没有主键')

        attr['mappings']=mappings
        attr['primary_key']=primary_key
        attr['table_name']=table_name
        return type.__new__(cls,name,bases,attr)

class Models(dict,metaclass=ModelsMeta):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        self[key]=value

    def __getattr__(self, item):
        try:
            return self[item]
        except BaseException:
            raise TypeError('没有这个属性')

    @classmethod
    def select_one(cls,**kwargs):
        key=list(kwargs.keys())[0]
        value=kwargs[key]
        sql='select * from %s where %s=?'%(cls.table_name,key)
        sql=sql.replace('?','%s')
        ms=mysql_conn.Mysql()
        re=ms.select(sql,value)
        if re:
            return cls(**re[0])
        else:
            return
    @classmethod
    def select_many(cls,**kwargs):
        ms=mysql_conn.Mysql()
        if kwargs:
            key = list(kwargs.keys())[0]
            value = kwargs[key]
            sql = 'select * from %s where %s=?' % (cls.table_name, key)
            sql = sql.replace('?', '%s')
            re = ms.select(sql, value)
        else:
            sql='select * from %s' %(cls.table_name)
            re = ms.select(sql, None)

        if re:
            return list(cls(**r) for r in re)
        else:
            return

    def update(self):
        ms=mysql_conn.Mysql()

        field_list=[]
        field_list_value=[]
        primary_key_value=None

        for k,v in self.mappings.items():
            if v.primary_key:
                primary_key_value=getattr(self,v.name,None)
            else:
                field_list.append(v.name+'=?')
                field_list_value.append(getattr(self,v.name,v.default))
        sql='update %s set %s where %s = %s'%(self.table_name,','.join(field_list),self.primary_key,primary_key_value)
        sql=sql.replace('?','%s')
        ms.execute(sql,field_list_value)

    def save(self):
        ms = mysql_conn.Mysql()

        field_list = []
        field_list_value = []
        char_list=[]

        for k, v in self.mappings.items():
            if not v.primary_key:
                field_list.append(v.name)
                char_list.append('?')
                field_list_value.append(getattr(self,v.name,v.default))
        sql='insert into %s(%s) value(%s)'%(self.table_name,','.join(field_list),','.join(char_list))
        sql=sql.replace('?','%s')
        ms.execute(sql,field_list_value)


class User(Models):
    table_name='user'
    id=IntegerField('id',primary_key=True)
    name=StringField('name')
    password=StringField('password')

if __name__ == '__main__':
    # user=User.select_one(id=1)
    # user.name='这测试1111'
    # user.update()
    # print(user)
    user=User(name='miaoqinian',password='xxx')
    user.save()



