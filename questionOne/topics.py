import csv
import json

f = open('./sample/answered_data_10k.in')
lines = f.readlines()
f.close()

f = csv.writer(open("train.csv", "wb+"))
g = csv.writer(open("train_topics.csv", "wb+"))

topics = []

f.writerow(["question_text", "followers", "name", "topicFollowers", "answer", "anonymous"])
g.writerow(["number of subtopics", "subtopic followers", "subtopic name"])
for x in lines:
    data = json.loads(x)
    try:
        topics = data['topics']
        if topics:
            tr = []
            tr.append(len(topics))            
            while topics:
                temp = topics.pop()
                tr.append(temp['followers'])
                tr.append(temp['name'])
        f.writerow([data["question_text"], 
	       data["context_topic"]["followers"], 
	       data["context_topic"]["name"], 
	       data["question_key"],
	       data["anonymous"],
	       data["__ans__"]])
        g.writerow(tr)
    except:
		pass