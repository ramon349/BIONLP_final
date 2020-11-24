import mongoengine

def global_init(name='ramen2'): 
    """ Handles creation of the  database. 
    """
    mongoengine.register_connection(alias="core",name=name) 