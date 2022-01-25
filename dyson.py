"""Syntax:
def glass = 玻璃
def ti = 钛
def water = 水
def ch4 = 可燃冰
def h = 氢
def gs = 光栅石
def cpu = 处理器
def tiglass = 钛化玻璃
def grap = 石墨烯
def kx = 卡西米尔晶体
def df = 位面过滤器
def qc = 量子芯片

machine made = 制造3, 1.5
machine che = 化工, 1

acc z3 = 增强剂3, 1.25, 0.01666667

process glass 2, ti 2, water 2 | made 5 | tiglass 2 @ z3
process ch4 2, h -1 | che 2 | grap 2 @ z3
process gs 4, grap 2, h 12 | made 4 | kx 1 @ z3
process tiglass 2, kx 1 | made 12 | df 1 @ z3
process df 2, cpu 2 | made 6 | qc 1 @ z3

source glass, ti, water, ch4, h, gs, cpu
product qc 1
"""

import typing as T
from dataclasses import dataclass
from collections import deque
import math
import sys


class Accelerator(T.NamedTuple):
    name: str
    acc_factor: float
    consume_factor: float


class Machine(T.NamedTuple):
    name: str
    time_factor: float


class Process(T.NamedTuple):
    materials: T.List[T.Tuple[str, float]]
    machine: str
    time: float
    produce_rate: float
    acc: T.Optional[str]


@dataclass
class Def:
    name: str
    is_source: bool = False
    is_product: bool = False
    product_need: float = 0
    process: T.Optional[int] = None

    visited: bool = False
    need: float = 0
    machine_count: int = 0
    max_rate_from_material: T.Optional[float] = 0  # None means infinite
    max_rate_from_machine: T.Optional[float] = 0  # None means infinite
    max_rate_from_consumer: T.Optional[float] = 0  # None means infinite
    actual_rate: float = 0
    state: int = 0
    # state == 1: max_rate_from_material is minimum, machine will wait for material (priority 1)
    # state == 2: max_rate_from_machine is minimum, machine will always be working (priority 0)
    # state == 3: max_rate_from_consumer is minimum, machine will stuck eventually (priority 2)


def strip_prefix(s: str, prefix: str) -> str:
    if not s.startswith(prefix):
        raise ValueError("`{}` does not start with `{}`".format(s, prefix))
    return s[len(prefix):].strip()


def strip_split(s: str, delim: str) -> T.List[str]:
    return [x.strip() for x in s.split(delim)]


def strip_split_2(s: str, delim: str, error_string: str) -> T.List[str]:
    ss = strip_split(s, delim)
    if len(ss) != 2:
        raise ValueError(error_string)
    return [ss[0], ss[1]]


def strip_split_3(s: str, delim1: str, delim2: str,
                  error_string: str) -> T.List[str]:
    if delim1 == delim2:
        ss = strip_split(s, delim1)
        if len(ss) != 3:
            raise ValueError(error_string)
        return [ss[0], ss[1], ss[2]]

    else:
        ss = strip_split(s, delim1)
        if len(ss) != 2:
            raise ValueError(error_string)
        ss2 = strip_split(ss[1], delim2)
        if len(ss2) != 2:
            raise ValueError(error_string)
        return [ss[0], ss2[0], ss2[1]]


def name_num_pair(s, error_string: str) -> T.Tuple[str, float]:
    s0, s1 = strip_split_2(s, " ", error_string)
    return s0, float(s1)


def min_with_none(*s: T.Tuple[T.Optional[float]]) -> T.Optional[float]:
    result = None
    for v in s:
        if v is None:
            continue
        elif result is None:
            result = v
        else:
            result = min(result, v)
    return result


class Dyson:
    def __init__(self):
        self._def = {}  # type: T.Dict[str, Def]
        self._machine = {}  # type: T.Dict[str, Machine]
        self._acc = {}  # type: T.Dict[str, Accelerator]
        self._process = []  # type: T.List[Process]

        self._successor = {}  # type: T.Dict[str, T.List[T.Tuple[str, float]]]
        self._real_source = []  # type: T.List[str]

    def eval_line(self, s: str, lineno: int):
        s = s.strip()

        if len(s) == 0:
            return

        elif s.startswith('def '):
            ss = strip_prefix(s, 'def ')
            sym, name = strip_split_2(
                ss, "=",
                "Syntax error (1) at line {}: bad material definition".format(
                    lineno))
            if sym in self._def:
                raise ValueError(
                    "Redefined error (2) at line {}: {} already defined".
                    format(lineno, sym))
            self._def[sym] = Def(name)

        elif s.startswith('source '):
            ss = strip_prefix(s, 'source ')
            syms = strip_split(ss, ",")
            for sym in syms:
                if not sym in self._def:
                    raise ValueError(
                        "Undefined error (3) at line {}: {} not defined".
                        format(lineno, sym))
                self._def[sym].is_source = True

        elif s.startswith('product '):
            ss = strip_prefix(s, 'product ')
            pairs = strip_split(ss, ",")
            for pair in pairs:
                sym, product_need = name_num_pair(
                    pair,
                    "Syntax error (4) at line {}: bad product definition".
                    format(lineno))
                if not sym in self._def:
                    raise ValueError(
                        "Undefined error (5) at line {}: {} not defined".
                        format(lineno, sym))
                self._def[sym].is_product = True
                self._def[sym].product_need = product_need

        elif s.startswith('machine '):
            ss = strip_prefix(s, 'machine ')
            sym, name, time_factor_str = strip_split_3(
                ss, "=", ",",
                "Syntax error (7) at line {}: bad machine definition".format(
                    lineno))
            time_factor = float(time_factor_str)
            if sym in self._machine:
                raise ValueError(
                    "Redefined error (8) at line {}: {} already defined".
                    format(lineno, sym))
            self._machine[sym] = Machine(name, time_factor)

        elif s.startswith('acc '):
            ss = strip_prefix(s, 'acc ')
            sym, parts = strip_split_2(
                ss, "=",
                "Syntax error (8) at line {}: bad accelerator definition".
                format(lineno))
            name, acc_factor_str, consume_factor_str = strip_split_3(
                parts, ",", ",",
                "Syntax error (9) at line {}: bad accelerator definition".
                format(lineno))
            acc_factor = float(acc_factor_str)
            consume_factor = float(consume_factor_str)
            if sym in self._acc:
                raise ValueError(
                    "Redefined error (10) at line {}: {} already defined".
                    format(lineno, sym))
            self._acc[sym] = Accelerator(name, acc_factor, consume_factor)

        elif s.startswith('process '):
            ss = strip_prefix(s, 'process ')
            material_part, machine_pair, produce_part = strip_split_3(
                ss, "|", "|",
                "Syntax error (11) at line {}: bad process definition".format(
                    lineno))

            material_pairs = strip_split(material_part, ",")
            materials = []
            for pair in material_pairs:
                sym, num = name_num_pair(
                    pair,
                    "Syntax error (12) at line {}: bad process definition".
                    format(lineno))
                if not sym in self._def:
                    raise ValueError(
                        "Undefined error (13) at line {}: {} not defined".
                        format(lineno, sym))
                materials.append((sym, num))

            machine_sym, time = name_num_pair(
                machine_pair,
                "Syntax error (14) at line {}: bad process definition".format(
                    lineno))
            if not machine_sym in self._machine:
                raise ValueError(
                    "Undefined error (15) at line {}: {} not defined".format(
                        lineno, machine_sym))

            acc_sym = None
            if '@' in produce_part:
                produce_part, acc_sym = strip_split_2(
                    produce_part, '@',
                    "Syntax error (16) at line {}: bad process definition".
                    format(lineno))
                if not acc_sym in self._acc:
                    raise ValueError(
                        "Undefined error (17) at line {}: {} not defined".
                        format(lineno, acc_sym))

            product_sym, produce_rate = name_num_pair(
                produce_part,
                "Syntax error (18) at line {}: bad process definition".format(
                    lineno))
            if not product_sym in self._def:
                raise ValueError(
                    "Undefined error (19) at line {}: {} not defined".format(
                        lineno, product_sym))
            if self._def[product_sym].process is not None:
                raise ValueError(
                    "Redefined error (20) at line {}: process of material {} is already defined"
                    .format(lineno, product_sym))
            process_index = len(self._process)
            self._def[product_sym].process = process_index
            self._process.append(
                Process(materials, machine_sym, time, produce_rate, acc_sym))

        else:
            raise ValueError(
                "Syntax error (21) at line {}: unknown command".format(lineno))

    def _calc_real_produce_rate(self, process: Process):
        real_produce_rate = process.produce_rate
        if process.acc is not None:
            acc = self._acc[process.acc]
            real_produce_rate *= acc.acc_factor
        return real_produce_rate

    def _check_loop(self):
        instack = set()  # type: T.Set[str]
        visited = set()  # type: T.Set[str]

        def _visitor(sym: str):
            if sym in visited:
                return
            instack.add(sym)
            visited.add(sym)

            def_ = self._def[sym]
            def_.visited = True
            if def_.process is None:
                instack.remove(sym)
                return

            process = self._process[def_.process]
            real_produce_rate = self._calc_real_produce_rate(process)

            for pred_sym, pred_need in process.materials:
                z = pred_need / real_produce_rate
                if pred_sym in instack:
                    raise ValueError(
                        "loop detected at symbol {}".format(pred_sym))
                if not pred_sym in self._successor:
                    self._successor[pred_sym] = []
                self._successor[pred_sym].append((sym, z))
                _visitor(pred_sym)

            instack.remove(sym)

        for sym, def_ in self._def.items():
            if def_.is_product:
                _visitor(sym)

    def _bfs_1(self):
        q = deque()  # type: T.Deque[str]
        successor_visited = {}  # type: T.Dict[str, int]

        for sym, def_ in self._def.items():
            if def_.is_product:
                def_.need = def_.product_need
                q.append(sym)

        while len(q) > 0:
            sym = q.popleft()
            def_ = self._def[sym]

            def_.max_rate_from_material = None

            if def_.is_source:
                def_.max_rate_from_machine = None
                self._real_source.append(sym)
                continue
            elif def_.process is None:
                raise ValueError("non source terminal: {}".format(sym))

            process = self._process[def_.process]
            real_produce_rate = self._calc_real_produce_rate(process)

            machine = self._machine[process.machine]
            real_time = process.time / machine.time_factor

            single_machine_rate = real_produce_rate / real_time
            def_.machine_count = int(math.ceil(def_.need /
                                               single_machine_rate))
            def_.max_rate_from_machine = def_.machine_count * single_machine_rate

            for pred_sym, pred_need in process.materials:
                pred_def = self._def[pred_sym]
                z = pred_need / real_produce_rate
                pred_def.need += def_.need * z

                if not pred_sym in successor_visited:
                    successor_visited[pred_sym] = 0
                successor_visited[pred_sym] += 1
                if successor_visited[pred_sym] == len(
                        self._successor[pred_sym]):
                    q.append(pred_sym)

    def _bfs_2(self):
        q = deque()  # type: T.Deque[str]
        predecessor_visited = {}  # type: T.Dict[str, int]

        for sym in self._real_source:
            q.append(sym)

        while len(q) > 0:
            sym = q.popleft()
            def_ = self._def[sym]

            if def_.is_product:
                continue

            max_p_rate = min_with_none(def_.max_rate_from_material,
                                       def_.max_rate_from_machine)

            for succ_sym, succ_z in self._successor[sym]:
                succ_def = self._def[succ_sym]
                if max_p_rate is not None:
                    succ_def.max_rate_from_material = min_with_none(
                        succ_def.max_rate_from_material, max_p_rate / succ_z)
                assert succ_def.process is not None, succ_sym
                process = self._process[succ_def.process]

                if not succ_sym in predecessor_visited:
                    predecessor_visited[succ_sym] = 0
                predecessor_visited[succ_sym] += 1
                if predecessor_visited[succ_sym] == len(process.materials):
                    q.append(succ_sym)

    def _bfs_3(self):
        q = deque()  # type: T.Deque[str]
        successor_visited = {}  # type: T.Dict[str, int]

        for sym, def_ in self._def.items():
            if def_.is_product:
                def_.max_rate_from_consumer = None
                q.append(sym)

        while len(q) > 0:
            sym = q.popleft()
            def_ = self._def[sym]

            def_.actual_rate = min_with_none(def_.max_rate_from_material,
                                             def_.max_rate_from_machine,
                                             def_.max_rate_from_consumer)

            if def_.actual_rate == def_.max_rate_from_machine:
                def_.state = 2
            elif def_.actual_rate == def_.max_rate_from_material:
                def_.state = 1
            else:
                def_.state = 3

            if def_.is_source:
                continue
            assert def_.process is not None, sym

            process = self._process[def_.process]
            real_produce_rate = self._calc_real_produce_rate(process)

            for pred_sym, pred_need in process.materials:
                pred_def = self._def[pred_sym]
                z = pred_need / real_produce_rate
                pred_def.max_rate_from_consumer += def_.actual_rate * z

                if not pred_sym in successor_visited:
                    successor_visited[pred_sym] = 0
                successor_visited[pred_sym] += 1
                if successor_visited[pred_sym] == len(
                        self._successor[pred_sym]):
                    q.append(pred_sym)

    def analyze(self):
        self._check_loop()
        self._bfs_1()
        self._bfs_2()
        self._bfs_3()

        # for sym, def_ in self._def.items():
        #     if not def_.visited:
        #         continue
        #     print("{}: {}".format(sym, def_))

        acc_point = {}
        acc_record = []
        sources = []
        products = []

        print("digraph g {")
        print('    rankdir = "LR"')

        for sym, def_ in self._def.items():
            if not def_.visited:
                continue
            label_parts = [def_.name, "{:.6g}/s".format(def_.actual_rate)]
            if def_.process is not None:
                process = self._process[def_.process]
                machine = self._machine[process.machine]
                label_parts.append("{}*{}".format(machine.name,
                                                  def_.machine_count))
            parts = ['label = "{}"'.format(", ".join(label_parts))]
            if def_.is_source:
                parts.append('shape = "octagon"')
                parts.append('style = "filled"')
                parts.append('fillcolor = "lightpink"')
                sources.append(sym)
            elif def_.is_product:
                parts.append('shape = "doubleoctagon"')
                parts.append('style = "filled"')
                parts.append('fillcolor = "lightskyblue"')
                products.append(sym)
            else:
                parts.append('shape = "box"')
            print('    {} [{}]'.format(sym, ", ".join(parts)))

        for sym, def_ in self._def.items():
            if not def_.visited:
                continue
            if def_.process is None:
                continue
            process = self._process[def_.process]

            real_produce_rate = self._calc_real_produce_rate(process)
            go_sum = 0
            for pred_sym, pred_need in process.materials:
                z = pred_need / real_produce_rate
                actual_go = def_.actual_rate * z
                print('    {} -> {} [label = "{:.6g}/s"]'.format(
                    pred_sym, sym, actual_go))
                if actual_go > 0:
                    go_sum += actual_go

            if process.acc is not None:
                acc = self._acc[process.acc]
                if not process.acc in acc_point:
                    acc_point[process.acc] = {"name": acc.name, "total": 0.0}
                acc_rate = go_sum * acc.consume_factor
                acc_record.append({
                    "acc": process.acc,
                    "product": sym,
                    "rate": acc_rate
                })
                acc_point[process.acc]["total"] += acc_rate

        for acc_sym, r in acc_point.items():
            parts = [r["name"], "{:.6g}/s".format(r["total"])]
            print(
                '    {} [label = "{}", shape = "box", style = "filled", fillcolor = "khaki"]'
                .format(acc_sym, ", ".join(parts)))

        for r in acc_record:
            print(
                '    {} -> {} [label = "{:.6g}/s", style = "dotted", color = "darkgoldenrod", fontcolor = "darkgoldenrod"]'
                .format(r["acc"], r["product"], r["rate"]))

        print('    {{ rank = "same"; {} }}'.format(", ".join(sources)))
        print('    {{ rank = "same"; {} }}'.format(", ".join(products)))

        print("}")


def main():
    dyson = Dyson()
    lineno = 1
    while True:
        input_ = sys.stdin.readline()
        if input_ == '':
            break
        dyson.eval_line(input_, lineno)
        lineno += 1

    dyson.analyze()


if __name__ == '__main__':
    main()
