import json
import tensorflow as tf


# top 50 subreddits in dataset, counted in DataProcessing.ipynb
top50subreddits = [
    'AskReddit',
    'relationships',
    'leagueoflegends',
    'tifu',
    'relationship_advice',
    'trees',
    'gaming',
    'atheism',
    'AdviceAnimals',
    'funny',
    'politics',
    'pics',
    'sex',
    'WTF',
    'explainlikeimfive',
    'todayilearned',
    'Fitness',
    'IAmA',
    'worldnews',
    'DotA2',
    'TwoXChromosomes',
    'videos',
    'DestinyTheGame',
    'reddit.com',
    'offmychest',
    'buildapc',
    'AskMen',
    'personalfinance',
    'summonerschool',
    'technology',
    'wow',
    'NoFap',
    'starcraft',
    'dating_advice',
    'askscience',
    'Games',
    'news',
    'talesfromtechsupport',
    'depression',
    'pcmasterrace',
    'Guildwars2',
    'magicTCG',
    'loseit',
    'GlobalOffensive',
    'electronic_cigarette',
    'movies',
    'self',
    'Advice',
    'Drugs',
    'teenagers'
]

top50subreddits_split = [
    'ask reddit',
    'relationships',
    'leagueoflegends',
    'tifu',
    'relationship advice',
    'trees',
    'gaming',
    'atheism',
    'advice animals',
    'funny',
    'politics',
    'pics',
    'sex',
    'WTF',
    'explain like im five',
    'today i learned',
    'fitness',
    'i am a',
    'world news',
    'DotA2',
    'two x chromosomes',
    'videos',
    'DestinyTheGame',
    'reddit.com',
    'off my chest',
    'build apc',
    'ask men',
    'personal finance',
    'summoner school',
    'technology',
    'wow',
    'no fap',
    'starcraft',
    'dating advice',
    'ask science',
    'games',
    'news',
    'tales from tech support',
    'depression',
    'pc master race',
    'Guildwars2',
    'magicTCG',
    'lose it',
    'GlobalOffensive',
    'electronic cigarette',
    'movies',
    'self',
    'advice',
    'drugs',
    'teenagers'
]

top50subreddits_splitcompound = {top50subreddits[i]: top50subreddits_split[i] for i in range(len(top50subreddits))}

def load_reddit_data_from_json():
    reddit_posts = []
    with open('../corpus-webis-tldr-17.json', 'r') as f:
        for i, line in enumerate(f):
            post = json.loads(line)
            del post['body']
            del post['normalizedBody']
            if 'subreddit' in post:
                reddit_posts.append(post)
            if i % 10**6 == 0:
                print(i)
    return reddit_posts


def extract_tensors(reddit_posts):
    subreddits = []
    content = []
    summary = []
    title = []
    raw_content = []

    while reddit_posts:
        post = reddit_posts.pop()
        if 'subreddit' in post and 'title' in post:
            subreddits.append(post['subreddit'])
            content.append(post['title'] +
                           ' in ' +
                           top50subreddits_splitcompound[post['subreddit']] +
                           '.\n' +
                           post['content'])
            title.append(post['title'])
            summary.append(post['summary'])
            raw_content.append(post['content'])

    subreddits = tf.convert_to_tensor(subreddits)
    content = tf.convert_to_tensor(content)
    summary = tf.convert_to_tensor(summary)
    title = tf.convert_to_tensor(title)
    raw_content = tf.convert_to_tensor(raw_content)
    return subreddits, content, summary, title, raw_content


# https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(topics, inputs, targets, titles, raw_inputs):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
      'topics': _bytes_feature(topics),
      'inputs':  _bytes_feature(inputs),
      'targets': _bytes_feature(targets),
      'titles': _bytes_feature(titles),
      'raw_inputs': _bytes_feature(raw_inputs),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(row):
    tf_string = tf.py_function(
        serialize_example,
        (row['topics'], row['inputs'], row['targets'], row['titles'], row['raw_inputs']),
        tf.string)
    return tf.reshape(tf_string, ())


def parser_fn(serialized_example):
    """Parse serialized examples."""
    features = tf.io.parse_single_example(
      serialized_example,
      features={
          "topics": tf.io.FixedLenFeature([], tf.string),
          "inputs": tf.io.FixedLenFeature([], tf.string),
          "targets": tf.io.FixedLenFeature([], tf.string),
          "titles": tf.io.FixedLenFeature([], tf.string),
          "raw_inputs": tf.io.FixedLenFeature([], tf.string),
      })
    return {
      "topics": features["topics"],
      "inputs": features["inputs"],
      "targets": features["targets"],
      "titles": features["titles"],
      "raw_inputs": features["raw_inputs"],
      "supervised": tf.constant(True)
    }


def build(input_pattern, shuffle_files):
    """Build dataset.

    Args:
      input_pattern: input file pattern.
      shuffle_files: whether to shuffle files list.

    Returns:
      Tuple of (tf.data.Dataset, number_of_examples)
    """
    filenames = sorted(tf.io.gfile.glob(input_pattern))
    if not filenames:
        raise ValueError("Can't not find files with pattern: %s." % input_pattern)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle_files:
        dataset = dataset.shuffle(len(filenames))
    options = tf.data.Options()
    options.experimental_deterministic = not shuffle_files
    dataset = dataset.with_options(options)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset