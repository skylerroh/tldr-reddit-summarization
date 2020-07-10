import json

def load_reddit_data_from_json():
    reddit_posts = []
    with open('corpus-webis-tldr-17.json', 'r') as f:
        for i, line in enumerate(f):
            post = json.loads(line)
            del post['body']
            del post['normalizedBody']
            reddit_posts.append(post)
            if i % 10**6 == 0:
                print(i)
    return reddit_posts