import psycopg2
import os

DB_NAME = os.environ.get("DB_NAME", "effihire")
DB_USER = os.environ.get("DB_USER", "admin")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")


def test_connection():
    """Attempts to connect to the PostgreSQL database and prints the status."""
    conn = None
    try:
        print(
            f"Attempting to connect to database '{DB_NAME}' on {DB_HOST}:{DB_PORT} as user '{DB_USER}'..."
        )
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            connect_timeout=5,  # Optional: Add a timeout
        )
        print("Connection successful!")

    except psycopg2.OperationalError as e:
        print(f"Connection failed: Unable to connect to the database.")
        print(f"Error details: {e}")
        print("\nTroubleshooting tips:")
        print("- Is the PostgreSQL server running? (sudo systemctl status postgresql)")
        print(
            f"- Are the connection details (host, port, dbname, user, password) correct?"
        )
        print(f"- Is the database '{DB_NAME}' created?")
        print(f"- Is the user '{DB_USER}' created and granted access?")
        print("- Is there a firewall blocking the connection?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn is not None:
            conn.close()
            print("Connection closed.")


if __name__ == "__main__":
    test_connection()
