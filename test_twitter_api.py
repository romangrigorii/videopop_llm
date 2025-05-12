import os
import tweepy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_twitter_api():
    # Get bearer token from environment variable
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        logger.error("Please set TWITTER_BEARER_TOKEN environment variable")
        return False
    
    try:
        # Initialize client
        client = tweepy.Client(bearer_token=bearer_token)
        
        # Test with a known account
        user = client.get_user(username='elonmusk')
        
        if user.data:
            logger.info(f"Successfully connected to Twitter API!")
            logger.info(f"Test user: {user.data.name} (@{user.data.username})")
            logger.info(f"Followers: {user.data.public_metrics['followers_count']:,}")
            return True
        else:
            logger.error("Failed to get user data")
            return False
            
    except Exception as e:
        logger.error(f"Error testing Twitter API: {str(e)}")
        return False

if __name__ == "__main__":
    test_twitter_api() 