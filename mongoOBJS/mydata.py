from enum import unique
import mongoengine 

#this is used by pymongo  generate documents in the datbase 
# object fields are also sotred in the datbaase 
#metadata is needed always 

class tweetData(mongoengine.Document):  
    tweetID=  mongoengine.StringField(required=True)
    text = mongoengine.StringField(required=True)
    report = mongoengine.StringField(required=True)
    meta = {
        'db_alias':'core',
        'collection':'samples'
    }