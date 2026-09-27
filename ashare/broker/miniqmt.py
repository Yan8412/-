"""Placeholder for a future Windows miniQMT / xtquant adapter.

Not constructed by the default CLI. Importing this module does not import
xtquant and does not send orders.
"""

from __future__ import annotations

from ashare.broker.base import AccountSnapshot, Broker, OrderAck, OrderRequest


class MiniQMTBroker(Broker):
    """Live adapter reserved for step 2.

    Programmatic trading needs a broker account with the量化/程序化 permission
    enabled, the miniQMT client on Windows, and the xtquant package that ships
    with that client. v1 raises immediately so a mis-set config cannot trade.
    """

    name = "miniqmt"

    def __init__(self) -> None:
        raise NotImplementedError(
            "v1 不连接券商，也不会下实盘单。miniQMT 适配留到下一阶段："
            "需 Windows、券商开通程序化交易权限，以及 xtquant。"
        )

    def place_orders(self, orders: list[OrderRequest]) -> list[OrderAck]:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> OrderAck:
        raise NotImplementedError

    def snapshot(self) -> AccountSnapshot:
        raise NotImplementedError
