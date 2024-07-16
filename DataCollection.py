import praw
from praw.models import MoreComments
# Import any other necessary packages from praw


class ImportData:
    # Hold a structure that contains all searchable sub-reddits
    def __init__(self):
        self.reddit = praw.Reddit(client_id='B_CviCpYficZIg',
                        client_secret='m1FpeRWtV0bUDkhzNlVzc55Rga0zjQ', user_agent='Programming Sentiment')
        self.subreddits = ['AskProgramming', 'learnprogramming', 'coding', 'CodingHelp', 'webdev', 'learnjava',
                           'learnjavascript', 'learnpython', 'Frontend', 'compsci']  # Add two more subreddits
        self.subreddit_data = dict(dict())

    def collect_data(self, sub):
        top_posts = self.reddit.subreddit(sub).top("all", limit=50)
        post_comments = dict()
        for submission in top_posts:
            comment_list = self.print_10_top_comments(submission)
            post_comments[submission.title] = comment_list
        self.subreddit_data[sub] = post_comments

    def print_10_top_comments(self, submission):
        # Set comment sort to best before retrieving comments
        submission.comment_sort = 'q&a'  # Gives comments where the author responded as the next response
        # Fetch the comments and print each comment body
        submission.comment_limit = 10
        # This must be done _after_ the above lines or they won't take affect.
        submission.comments.replace_more(limit=0)
        comment_list = []
        for comment in submission.comments.list():
            comment_list.append(comment.body)
        return comment_list

    def subreddit_scrape(self):
        for i, sub in enumerate(self.subreddits):
            self.collect_data(sub)
            print(f"sub-reddits completed {i}")

    def to_file(self, file):
        with open(file, 'w', encoding='utf-8') as f:
            for subreddit in self.subreddit_data:
                for post in self.subreddit_data[subreddit]:
                    post = post.strip()
                    post_line = "P: " + post + "\n"
                    f.write(post_line)
                    for comment in self.subreddit_data[subreddit][post]:
                        comment = comment.split()
                        comment = [word.strip() for word in comment]
                        comment = " ".join(comment)
                        comment_line = "C: " + comment + "\n"
                        f.write(comment_line)


if __name__ == "__main__":
    collector = ImportData()
    collector.subreddit_scrape()
    collector.to_file("RedditAnnotation.txt")

