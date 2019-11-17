import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate(u".account")
firebase_admin.initialize_app(cred)

db = firestore.client()



# doc = db.document(u'music/R119zKhuB6PeYJR8MxP4JvWuEg43/musicId/04zZlFjdpyvzXy7WjbGo').get()

# docs = db.collection(u'music').where(u'id', u'==', u'R119zKhuB6PeYJR8MxP4JvWuEg43').stream()


docs = db.collection(u'music/R119zKhuB6PeYJR8MxP4JvWuEg43/musicId').where(u'filename', u'==', u'recorded-8c3e82a6-972d-4dd0-910e-a7d4c919742c.wav').stream()

# docs = db.collection(u'music').where(u'id', u'==', u'R119zKhuB6PeYJR8MxP4JvWuEg43').stream()

# print(u'{} => {}'.format(doc.id, doc.to_dict()))

for doc in docs:
    print(u'{} => {}'.format(doc.id, doc.to_dict()))
