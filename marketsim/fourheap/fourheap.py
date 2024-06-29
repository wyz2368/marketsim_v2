from collections import defaultdict
from marketsim.fourheap import constants
from marketsim.fourheap.order import Order
from marketsim.fourheap.order_queue import OrderQueue

import math
import numpy as np


class FourHeap:
    """
    This class reimplements the four-heap data structure described in "Flexible double auctions for electronic commerce:
    theory and implementation" (Wurman, 98)
    """
    def __init__(self, plus_one=False):
        self.plus_one = plus_one

        self.buy_matched = OrderQueue(is_max_heap=False, is_matched=True)
        self.buy_unmatched = OrderQueue(is_max_heap=True, is_matched=False)
        self.sell_matched = OrderQueue(is_max_heap=True, is_matched=True)
        self.sell_unmatched = OrderQueue(is_max_heap=False, is_matched=False)

        self.heaps = [self.buy_matched, self.buy_unmatched, self.sell_matched, self.sell_unmatched]
        self.agent_id_map = defaultdict(list)

        self.midprices = []

    def handle_new_order(self, order):
        q_order = order.quantity
        order_matched = self.sell_matched if order.order_type == constants.SELL else self.buy_matched
        counter_matched = self.sell_matched if order.order_type == constants.BUY else self.buy_matched
        counter_unmatched = self.sell_unmatched if order.order_type == constants.BUY else self.buy_unmatched

        to_match = counter_unmatched.push_to()
        if to_match is not None:
            to_match_quantity = to_match.quantity
            if to_match_quantity == q_order:
                order_matched.add_order(order)
                counter_matched.add_order(to_match)
            elif to_match_quantity > q_order:
                excess_order = to_match.copy_and_decrease(q_order)
                order_matched.add_order(order)
                counter_matched.add_order(to_match)
                counter_unmatched.add_order(excess_order)
            elif q_order > to_match_quantity:
                # There's a better way to do this, but I think it's not worth it
                counter_matched.add_order(to_match)
                new_order = order.copy_and_decrease(to_match_quantity)
                order_matched.add_order(order)
                self.insert(new_order)

    def handle_replace(self, order):
        matched = self.sell_matched if order.order_type == constants.SELL else self.buy_matched
        unmatched = self.sell_unmatched if order.order_type == constants.SELL else self.buy_unmatched
        q_order = order.quantity
        replaced = matched.push_to()
        if replaced is not None:
            replaced_quantity = replaced.quantity
            if replaced_quantity == q_order:
                matched.add_order(order)
                unmatched.add_order(replaced)
            elif replaced_quantity > q_order:
                matched.add_order(order)
                matched_s = replaced.copy_and_decrease(q_order)
                matched.add_order(matched_s)
                unmatched.add_order(replaced)
            elif replaced_quantity < q_order:
                new_order = order.copy_and_decrease(replaced_quantity)
                matched.add_order(order)
                unmatched.add_order(replaced)
                self.insert(new_order)

    def insert(self, order: Order):
        self.agent_id_map[order.agent_id].append(order.order_id)
        if order.order_type == constants.SELL:
            if order.price <= self.buy_unmatched.peek() and self.sell_matched.peek() <= self.buy_unmatched.peek():
                self.handle_new_order(order)
            elif order.price <= self.sell_matched.peek():
                self.handle_replace(order)
            else:
                self.sell_unmatched.add_order(order)
        elif order.order_type == constants.BUY:
            if order.price >= self.sell_unmatched.peek() and self.buy_matched.peek() >= self.sell_unmatched.peek():
                self.handle_new_order(order)
            elif order.price >= self.buy_matched.peek():
                self.handle_replace(order)
            else:
                self.buy_unmatched.add_order(order)

    def remove(self, order_id: int):
        if self.buy_unmatched.contains(order_id):
            self.buy_unmatched.remove(order_id)
        elif self.sell_unmatched.contains(order_id):
            self.sell_unmatched.remove(order_id)
        elif self.buy_matched.contains(order_id):
            order_q = self.buy_matched.order_dict[order_id].quantity
            self.buy_matched.remove(order_id)
            s = self.sell_matched.push_to()
            s_quantity = s.quantity
            if s_quantity == order_q:
                self.insert(s)
            elif s_quantity > order_q:
                diff = s_quantity - order_q
                s.quantity -= diff
                self.insert(s)
                self.sell_matched.order_dict[s.order_id].quantity += diff
            elif s_quantity < order_q:
                while order_q > 0:
                    order_q -= s_quantity
                    self.insert(s)
                    s = self.sell_matched.push_to()
                    s_quantity = s.quantity
        elif self.sell_matched.contains(order_id):
            order_q = self.sell_matched.order_dict[order_id].quantity
            self.sell_matched.remove(order_id)
            b = self.buy_matched.push_to()
            b_quantity = b.quantity
            if b_quantity == order_q:
                self.insert(b)
            elif b_quantity > order_q:
                diff = b_quantity - order_q
                b.quantity -= diff
                self.insert(b)
                self.buy_matched.order_dict[b.order_id].quantity += diff
            elif b_quantity < order_q:
                while order_q > 0:
                    order_q -= b_quantity
                    self.insert(b)
                    b = self.buy_matched.push_to()
                    b_quantity = b.quantity

    def withdraw_all(self, agent_id: int):
        for order_id in self.agent_id_map[agent_id]:
            self.remove(order_id)
        self.agent_id_map[agent_id] = []

    def market_clear(self, t):
        p = self.get_ask_quote() if self.plus_one else self.get_bid_quote()

        buy_matched = self.buy_matched.market_clear(p,t)
        sell_matched = self.sell_matched.market_clear(p,t)

        matched_orders = buy_matched + sell_matched
        return matched_orders

    def get_bid_quote(self) -> float:
        return max(self.buy_unmatched.peek(), self.sell_matched.peek())

    def get_ask_quote(self) -> float:
        return max(self.sell_unmatched.peek(), self.buy_matched.peek())

    def get_best_bid(self) -> float:
        return self.buy_unmatched.peek()

    def get_best_ask(self) -> float:
        return self.sell_unmatched.peek()

    def update_midprice(self, lookback=14):
        best_ask = self.get_best_ask()
        best_bid = self.get_best_bid()

        if math.isinf(best_ask) or math.isinf(best_bid):
            if len(self.midprices) < lookback and len(self.midprices) > 0:
                self.midprices.append(np.mean(self.midprices))
            elif len(self.midprices) >= lookback:
                self.midprices.append(np.mean(self.midprices[-lookback:]))
        else:
            self.midprices.append((best_ask + best_bid) / 2)



    def observe(self) -> str:
        s = '--------------\n'
        names = ['buy_matched', 'buy_unmatched', 'sell_matched', 'sell_unmatched']
        for i, heap in enumerate(self.heaps):
            s += names[i]
            s += '\n'
            # s += f'Top order_id: {heap.peek_order().order_id}\n'
            s += f'Top price: {abs(heap.peek())}\n'
            s += f'Number of orders: {heap.count()}\n\n\n'

        return s
