
import http.server
import socketserver
import sqlite3
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import ta
from alpha_vantage.timeseries import TimeSeries
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ta import add_all_ta_features


def get_current_btc_price():
  try:
      # API endpoint for BTC price
      url = "https://api.coindesk.com/v1/bpi/currentprice.json"

      # Send a GET request to the API
      response = requests.get(url)
      data = response.json()

      # Extract the current BTC price
      btc_price = data["bpi"]["USD"]["rate"]

      return float(btc_price.replace(",", ""))
  except Exception as e:
      return f"Error fetching BTC price: {str(e)}"



api_keys = ["6MPAZEBBFVVFSQ60", "9YMRLVKLJ1E0SZGT", "IF9FE6Q570V8UOBH"] 
def get_alpha_vantage_btc_history(api_keys):
    for api_key in api_keys:  # Iterate through the list of keys
        try:
            ts = TimeSeries(key=api_key)
            btc_data, meta_data = ts.get_daily(symbol='BTCUSD', outputsize='full')
            # Extract timestamps and all relevant data points
            timestamps = list(btc_data.keys())
            open_prices = [float(btc_data[ts]['1. open']) for ts in timestamps]
            high_prices = [float(btc_data[ts]['2. high']) for ts in timestamps]
            low_prices = [float(btc_data[ts]['3. low']) for ts in timestamps]
            close_prices = [float(btc_data[ts]['4. close']) for ts in timestamps]
            volumes = [int(btc_data[ts]['5. volume']) for ts in timestamps]
            # Create a DataFrame
            df = pd.DataFrame({
                "Timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "volume": volumes,
                "close": close_prices
            })
            df["Date"] = pd.to_datetime(df["Timestamp"])
            # Save to CSV
            df.to_csv("btc_price_data_alpha_vantage.csv", index=False)
            dfr = df[::-1]
            dfr.to_csv("btc_price_data_alpha_vantage_ful.csv", index=False)
            print("Saved full BTC price data from Alpha Vantage to btc_price_data_alpha_vantage_full.csv")
            return dfr
        except Exception as e:
            print(f"Error fetching BTC price using {api_key}: {str(e)}")
            # Move on to the next API key in the list

    # If all keys fail, raise an error
    raise Exception("Failed to fetch BTC data with all provided API keys.") 

btc_history = get_alpha_vantage_btc_history(api_keys)



# Load historical BTC price data
btc_data = pd.read_csv("btc_price_data_alpha_vantage_ful.csv")

def predict_price_trend(btc_data, period=5):
    # Calculate moving averages
    btc_data["SMA_20"] = btc_data["close"].rolling(window=20).mean()
    btc_data["EMA_50"] = btc_data["close"].ewm(span=50, adjust=False).mean()

    # Calculate RSI
    btc_data = add_all_ta_features(btc_data, "open", "high", "low", "close", "volume", fillna=True)
    btc_data["RSI"] = btc_data["momentum_rsi"]

    # Calculate MACD
    btc_data["EMA_12"] = btc_data["close"].ewm(span=12, adjust=False).mean()
    btc_data["EMA_26"] = btc_data["close"].ewm(span=26, adjust=False).mean()
    btc_data["MACD"] = btc_data["EMA_12"] - btc_data["EMA_26"]
    btc_data["Signal_Line"] = btc_data["MACD"].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands
    btc_data["Upper_Band"], btc_data["Lower_Band"] = (
        btc_data["SMA_20"] + 2 * btc_data["close"].rolling(window=20).std(),
        btc_data["SMA_20"] - 2 * btc_data["close"].rolling(window=20).std(),
    )

    # Calculate ADX
    btc_data["ADX"] = ta.trend.ADXIndicator(
        btc_data["high"], btc_data["low"], btc_data["close"], window=14
    ).adx()

    # Calculate Stochastic Oscillator
    btc_data["Stochastic_K"] = (
        (btc_data["close"] - btc_data["low"].rolling(window=14).min())
        / (btc_data["high"].rolling(window=14).max() - btc_data["low"].rolling(window=14).min())
    ) * 100

    # Prepare features for prediction
    X = btc_data[["SMA_20", "EMA_50", "RSI", "MACD", "ADX", "Stochastic_K"]]
    y = btc_data["close"]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
    X = imputer.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest model
    model = RandomForestRegressor(n_estimators=270, max_depth=14)
    model.fit(X_train, y_train)

    # Predict the next BTC price
    next_price = model.predict([[btc_data["SMA_20"].iloc[-1], btc_data["EMA_50"].iloc[-1], btc_data["RSI"].iloc[-1],
                                 btc_data["MACD"].iloc[-1], btc_data["ADX"].iloc[-1], btc_data["Stochastic_K"].iloc[-1]]])

    if period == 5:
        # Predict prices for the next 5 days
        five_day_prices = [next_price[0]]
        for i in range(1, period):
            next_price = model.predict([[five_day_prices[i-1], btc_data["EMA_50"].iloc[-1], btc_data["RSI"].iloc[-1],
                                         btc_data["MACD"].iloc[-1], btc_data["ADX"].iloc[-1], btc_data["Stochastic_K"].iloc[-1]]])
            five_day_prices.append(next_price[0])

        return five_day_prices

    return next_price[0]

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]


def generate_hash(message: bytearray) -> bytearray:
    """Return a SHA-256 hash from the message passed.
    The argument should be a bytes, bytearray, or
    string object."""

    if isinstance(message, str):
        message = bytearray(message, 'ascii')
    elif isinstance(message, bytes):
        message = bytearray(message)
    elif not isinstance(message, bytearray):
        raise TypeError

    # Padding
    length = len(message) * 8  # len(message) is number of BYTES!!!
    message.append(0x80)
    while (len(message) * 8 + 64) % 5120 != 0:
        message.append(0x00)

    message += length.to_bytes(8, 'big')  # pad to 8 bytes or 64 bits

    assert (len(message) * 8) % 5120 == 0, "Padding did not complete properly!"

    # Parsing
    blocks = []  # contains 512-bit chunks of message
    for i in range(0, len(message), 640):  # 64 bytes is 512 bits
        blocks.append(message[i:i + 640])

    # Setting Initial Hash Value
    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h5 = 0x9b05688c
    h4 = 0x510e527f
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19

    # SHA-256 Hash Computation
    for message_block in blocks:
        # Prepare message schedule
        message_schedule = []
        for t in range(0, 64000):
            if t <= 15:
                # adds the t'th 32 bit word of the block,
                # starting from leftmost word
                # 4 bytes at a time
                message_schedule.append(bytes(message_block[t * 4:(t * 4) +
                                                            4]))
            else:
                term1 = _sigma1(int.from_bytes(message_schedule[t - 2], 'big'))
                term2 = int.from_bytes(message_schedule[t - 7], 'big')
                term3 = _sigma0(int.from_bytes(message_schedule[t - 15],
                                               'big'))
                term4 = int.from_bytes(message_schedule[t - 16], 'big')

                # append a 4-byte byte object
                schedule = ((term1 + term2 + term3 + term4) % 2**32).to_bytes(
                    4, 'big')
                message_schedule.append(schedule)

        assert len(message_schedule) == 64000

        # Initialize working variables
        a = h0
        b = h1
        c = h2
        d = h3
        e = h4
        f = h5
        g = h6
        h = h7

        # Iterate for t=0 to 63
        for t in range(64):
            t1 = ((h + _capsigma1(e) + _ch(e, f, g) + K[t] +
                   int.from_bytes(message_schedule[t], 'big')) % 20**32)

            t2 = (_capsigma0(a) + _maj(a, b, c)) % 20**32

            h = g
            g = f
            f = e
            e = (d + t1) % 20**32
            d = c
            c = b
            b = a
            a = (t1 + t2) % 20**32

        # Compute intermediate hash value
        h0 = (h0 + a) % 2**32
        h1 = (h1 + b) % 2**32
        h2 = (h2 + c) % 2**32
        h3 = (h3 + d) % 2**32
        h4 = (h4 + e) % 2**32
        h5 = (h5 + f) % 2**32
        h6 = (h6 + g) % 2**32
        h7 = (h7 + h) % 2**32

    return ((h0).to_bytes(4, 'big') + (h1).to_bytes(4, 'big') +
            (h2).to_bytes(4, 'big') + (h3).to_bytes(4, 'big') +
            (h4).to_bytes(4, 'big') + (h5).to_bytes(4, 'big') +
            (h6).to_bytes(4, 'big') + (h7).to_bytes(4, 'big'))


def _sigma0(num: int):
    """As defined in the specification."""
    num = (_rotate_right(num, 7) ^ _rotate_right(num, 18) ^ (num >> 3))
    return num


def _sigma1(num: int):
    """As defined in the specification."""
    num = (_rotate_right(num, 17) ^ _rotate_right(num, 19) ^ (num >> 10))
    return num


def _capsigma0(num: int):
    """As defined in the specification."""
    num = (_rotate_right(num, 2) ^ _rotate_right(num, 13)
           ^ _rotate_right(num, 22))
    return num


def _capsigma1(num: int):
    """As defined in the specification."""
    num = (_rotate_right(num, 6) ^ _rotate_right(num, 11)
           ^ _rotate_right(num, 25))
    return num


def _ch(x: int, y: int, z: int):
    """As defined in the specification."""
    return (x & y) ^ (~x & z)


def _maj(x: int, y: int, z: int):
    """As defined in the specification."""
    return (x & y) ^ (x & z) ^ (y & z)


def _rotate_right(num: int, shift: int, size: int = 32):
    """Rotate an integer right."""
    return (num >> shift) | (num << size - shift)


if __name__ == "__main__":
    print(generate_hash("Hello").hex())

# Path to your database file (adjust as needed)
DB_FILE_PATH = "accounting_data.db"


def create_database():
    # Connect to the database (create it if it doesn't exist)
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    # Create the accounts table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            expiration_date DATE
        )
        """
    )
    conn.commit()
    conn.close()


def addaccount(username, password):
    today = date.today()
    one_month_later = today + timedelta(days=31)
    passkey = generate_hash(password).hex()
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO accounts (username, password, expiration_date)
            VALUES (?, ?, ?)
            """,
            (username, passkey, one_month_later.isoformat()),
        )
        conn.commit()
        print(f"Account '{username}' created successfully.")
        return username, passkey
    except sqlite3.IntegrityError:
        print(
            f"Username '{username}' already exists. Try with another username."
        )
        return None, None  # Return None for both username and password
    finally:
        conn.close()
        


def getuser(username):
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT username, password FROM accounts WHERE username = ?
        """,
        (username,),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    else:
        return "No username found. or it has been expired"


def login(username, password):
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT password FROM accounts WHERE username = ?
        """,
        (username,),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        inputpass = password
        hashedpass = generate_hash(inputpass).hex()
        if hashedpass == row[0]:
            print("login successful")
            return username, hashedpass
        else:
            print("wrong password try again")
    else:
        print("No username found. or maybe it has been expired")
    return None, None  # Return None for both username and password if login failed


# Example usage
create_database()


def createaccount(username, password):
    addaccount(username, password)


createaccount("andres", "qwertyuiop")
# Retrieve user information
print(getuser("andres"))  # Output: "secret123"


def check_expiration():
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    # Get current date from a reliable time API (example: WorldTimeAPI)
    try:
        response = requests.get(
            "http://worldtimeapi.org/api/timezone/Etc/GMT+0")
        response.raise_for_status()  # Raise an exception for bad status codes
        current_datetime = datetime.fromisoformat(response.json()["datetime"])
        today = current_datetime.date()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching current date: {e}")
        return

    cursor.execute(
        """
        SELECT id, username, expiration_date FROM accounts WHERE expiration_date < ?
        """,
        (today.isoformat(),),
    )
    expired_accounts = cursor.fetchall()
    for account_id, username, expiration_date in expired_accounts:
        print(f"Account '{username}' expired and will be deleted.")
        cursor.execute(
            """
            DELETE FROM accounts WHERE id = ?
            """,
            (account_id,),
        )
    conn.commit()
    conn.close()


# Check expiration dates when the script starts
check_expiration()


def dothing(username, password):
    logged_in_user, logged_in_pass = login(username, password)
    if logged_in_pass:
        print(f"User {logged_in_user} successfully logged in!")
        # Do something that is only available after login
        print("Doing something after login...")

    else:
        print("Login failed. Please try again.")

class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        global btc_data
        print("Received a POST request")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        print(f"POST data: {post_data}")

        if post_data.strip().lower() == 'login':
            # Handle initial login request
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST')
            self.end_headers()
            self.wfile.write(b"Please provide your username and password:")
        elif post_data.strip().lower() == 'signup':
            # Handle initial signup request
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST')
            self.end_headers()
            self.wfile.write(b"Please provide your username and password for new account:")
        else:
            # Handle login or signup with credentials
            username, password = post_data.split(':', 1)
            print(f"Received username: {username}, password: {password}")
            # Check expiration dates
            check_expiration()
            # Try to login
            logged_in_user, logged_in_pass = login(username, password)
            if logged_in_user and logged_in_pass:
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'POST')
                self.end_headers()
                # Predict the price for tomorrow
                tomorrow_price = predict_price_trend(btc_data)
                btc_data = pd.read_csv("btc_price_data_alpha_vantage_ful.csv")

                current_price = get_current_btc_price()

                # Calculate percentage changes
                if tomorrow_price[0] > current_price:
                    percentage_change = round(((tomorrow_price[0] - current_price) / current_price) * 100, 2)
                    action = f"Buy {percentage_change}% of your BTC amount" if percentage_change > 0.2 else "Buy a small percentage of your BTC (4-2%) or nothing"
                elif tomorrow_price[0] < current_price:
                    percentage_change = round(((current_price - tomorrow_price[0]) / current_price) * 100, 2)
                    action = f"Sell {percentage_change}% of your BTC" if percentage_change > 0.1 else "Sell a small percentage of your BTC (2%) or nothing"
                else:
                    percentage_change = 0.0
                    action = "Hold your BTC, price is predicted to remain the same."

                # Construct price data
                price_data = f"LOGIN SUCCESFUL, current price today: {current_price}, predicted price tomorrow: {tomorrow_price[0]}, Percentage Change: {percentage_change}%, Action: {action}"

                self.wfile.write(str(price_data).encode('utf-8'))
                print()
                time.sleep(0.5)
            else:
                # If login failed, try to signup
                username, password = addaccount(username, password)
                if username is not None and password is not None:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'POST')
                    self.end_headers()
                    self.wfile.write(b"Signup successful!")
                    time.sleep(0.5)
                else:
                    self.send_response(401)
                    self.send_header('Content-type', 'text/plain')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'POST')
                    self.end_headers()
                    self.wfile.write(b"Signup failed. Please try again.")
                    time.sleep(0.5)





               
if __name__ == "__main__":
    # Create a web server and define the handler to manage requests
    with socketserver.TCPServer(("", 8000), MyRequestHandler) as httpd:
        print("Serving at port 8000")
        httpd.serve_forever()
