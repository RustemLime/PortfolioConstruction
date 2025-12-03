import pprint
from lime_trader import LimeClient

from _decimal import Decimal

from lime_trader.models.trading import Order, TimeInForce, OrderType, OrderSide
import uuid
import json


# # Get the directory where this script is located
# script_dir = Path(__file__).parent
# credentials_path = script_dir / "credentials.json"
# client = LimeClient.from_file(file_path=str(credentials_path))

client = LimeClient.from_json("""{
	"username": "hakaca7421@docsfy.com",
	"password": "123Qwe123",
	"client_id": "trading-app-dmo04416",
	"client_secret": "04d7124d23084a20a87e4443f22eba53",
	"grant_type": "password",
	"base_url": "https://api.lime.co",
	"auth_url": "https://auth.lime.co"
}""")

def get_balances():
    """
    This function returns the balances of the account.
    Returns:
    balances: dictionary of balances
    """
    balances = client.account.get_balances()
    return balances

def get_positions():
    """
    This function returns the positions of the account.
    Returns:
    positions: dictionary of positions
    """
    accounts = client.account.get_balances()  # need to get account numbers first

    account_number = accounts[0].account_number # get account number of first account in a list

    positions = client.account.get_positions(account_number=account_number)
    return positions

def send_order(symbol, quantity, side = "buy"):
    """
    This function sends an order to the exchange.
    Parameters:
    symbol: symbol of the stock
    quantity: quantity of the stock
    side: side of the order
    Returns:
    placed_order_response: dictionary of the placed order response
    Example:
    placed_order_response = send_order(symbol='AAPL', quantity=2, side='sell')
    placed_order_response = send_order(symbol='QQQ', quantity=3, side='buy')

    """
    accounts = client.account.get_balances()  # need to get account numbers first

    account_number = accounts[0].account_number  # get account number of first account in a list
    client_order_id = str(uuid.uuid4()).replace('-', '')[:32] # limit client order id to 32 characters, remove hyphens
    side = OrderSide.BUY if side == 'buy' else OrderSide.SELL

    order = Order(account_number=account_number,
                symbol=symbol,
                quantity=Decimal(quantity),
                order_type=OrderType.MARKET,
                side=side,
                client_order_id=client_order_id,
                )
    placed_order_response = client.trading.place_order(order=order)
    return placed_order_response

# print(get_balances())
# print(send_order(symbol='AAPL', quantity=2, side='buy'))
# print(get_positions())
