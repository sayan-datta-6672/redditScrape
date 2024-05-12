# importing libraries
import os
import asyncpraw
import pandas as pd
import aiohttp
import nest_asyncio
import asyncio
import csv

nest_asyncio.apply()

def save_to_csv(comments):
    with open('output.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(comments)

# defining a function scrape which will take the search query as input and
# output the resulting posts and their respective comments/replies into a csv file
async def scrape(search_query):
    # creating a custom session
    conn = aiohttp.TCPConnector(ssl=False)
    session = aiohttp.ClientSession(connector=conn)
    # instantiating the reddit object with the keys
    reddit = asyncpraw.Reddit(
        client_id=os.environ['reddit_client_id'],  # your client id
        client_secret=os.environ['reddit_client_secret'],  # your client secret
        user_agent=os.environ['user_agent'],  # your user agent
        requestor_kwargs={"session": session}  # session
    )
    sort = 'relevance'  # "relevance", "hot", "top", "new", "comments"
    syntax = 'lucene'  # "cloudsearch", "lucene", or "plain"
    time_filter = 'all'  # "all", "day", "hour", "month", "week", "year"

    try:
        subreddit = await reddit.subreddit("all")
        search_results_count = 0
        post_comments = []
        async for submission in subreddit.search(search_query, sort, syntax, time_filter):
            if submission.ups > 10:
                search_results_count += 1
                subreddit = submission.subreddit
                print("Processing subreddit: ", search_results_count, ".", subreddit)
                if subreddit is not None:
                    comment_submission = await reddit.submission(id=submission.id)
                    comments = comment_submission.comments
                    await comments.replace_more(limit=None)
                    comment_queue = comments[:]  # Seed with top-level
                    while comment_queue:
                        comment = comment_queue.pop(0)
                        comment_queue.extend(comment.replies)
                        post_comments.append(comment.body)
                else:
                    print("Subreddit is None")

        return post_comments
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await reddit.close()


async def main(query):
    response = await scrape(query)
    # print(response)
    # save_to_csv(response)
    comments_df = pd.DataFrame(response)
    comments_df.to_csv(f'{query}.csv')
    # print("Total search results: ", response['search_result_count'])
    # print("Total number of comments: ", comments_df.size)


if __name__ == "__main__":
    query = input("Search for: ")
    asyncio.run(main(query))
