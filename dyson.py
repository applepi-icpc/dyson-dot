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
    succ_output_rate: float = 0
    state: int = 0
    # state == 1: max_rate_from_material is minimum, machine will wait for material (priority 1)
    # state == 2: max_rate_from_machine is minimum, machine will always be working (priority 0)
    # state == 3: max_rate_from_consumer is minimum, machine will stuck eventually (priority 2)


def state_str(state: int):
    if state == 0:
        return "unknown"
    elif state == 1:
        return "wait_material"
    elif state == 2:
        return "always_running"
    elif state == 3:
        return "wait_consumption"
    else:
        return "invalid"


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

        elif s.startswith('#'):
            # comment
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
                if self._def[sym].is_product:
                    raise ValueError(
                        "Conflict error (4) at line {}: {} is already a product"
                        .format(lineno, sym))
                self._def[sym].is_source = True

        elif s.startswith('product '):
            ss = strip_prefix(s, 'product ')
            pairs = strip_split(ss, ",")
            for pair in pairs:
                sym, product_need = name_num_pair(
                    pair,
                    "Syntax error (5) at line {}: bad product definition".
                    format(lineno))
                if not sym in self._def:
                    raise ValueError(
                        "Undefined error (6) at line {}: {} not defined".
                        format(lineno, sym))
                if self._def[sym].is_source:
                    raise ValueError(
                        "Conflict error (7) at line {}: {} is already a source"
                        .format(lineno, sym))
                self._def[sym].is_product = True
                self._def[sym].product_need = product_need

        elif s.startswith('machine '):
            ss = strip_prefix(s, 'machine ')
            sym, name, time_factor_str = strip_split_3(
                ss, "=", ",",
                "Syntax error (8) at line {}: bad machine definition".format(
                    lineno))
            time_factor = float(time_factor_str)
            if sym in self._machine:
                raise ValueError(
                    "Redefined error (9) at line {}: {} already defined".
                    format(lineno, sym))
            self._machine[sym] = Machine(name, time_factor)

        elif s.startswith('acc '):
            ss = strip_prefix(s, 'acc ')
            sym, parts = strip_split_2(
                ss, "=",
                "Syntax error (10) at line {}: bad accelerator definition".
                format(lineno))
            name, acc_factor_str, consume_factor_str = strip_split_3(
                parts, ",", ",",
                "Syntax error (11) at line {}: bad accelerator definition".
                format(lineno))
            acc_factor = float(acc_factor_str)
            consume_factor = float(consume_factor_str)
            if sym in self._acc:
                raise ValueError(
                    "Redefined error (12) at line {}: {} already defined".
                    format(lineno, sym))
            self._acc[sym] = Accelerator(name, acc_factor, consume_factor)

        elif s.startswith('process '):
            ss = strip_prefix(s, 'process ')
            material_part, machine_pair, produce_part = strip_split_3(
                ss, "|", "|",
                "Syntax error (13) at line {}: bad process definition".format(
                    lineno))

            material_pairs = strip_split(material_part, ",")
            materials = []
            for pair in material_pairs:
                sym, num = name_num_pair(
                    pair,
                    "Syntax error (14) at line {}: bad process definition".
                    format(lineno))
                if not sym in self._def:
                    raise ValueError(
                        "Undefined error (15) at line {}: {} not defined".
                        format(lineno, sym))
                materials.append((sym, num))

            machine_sym, time = name_num_pair(
                machine_pair,
                "Syntax error (16) at line {}: bad process definition".format(
                    lineno))
            if not machine_sym in self._machine:
                raise ValueError(
                    "Undefined error (17) at line {}: {} not defined".format(
                        lineno, machine_sym))

            acc_sym = None
            if '@' in produce_part:
                produce_part, acc_sym = strip_split_2(
                    produce_part, '@',
                    "Syntax error (18) at line {}: bad process definition".
                    format(lineno))
                if not acc_sym in self._acc:
                    raise ValueError(
                        "Undefined error (19) at line {}: {} not defined".
                        format(lineno, acc_sym))

            product_sym, produce_rate = name_num_pair(
                produce_part,
                "Syntax error (20) at line {}: bad process definition".format(
                    lineno))
            if not product_sym in self._def:
                raise ValueError(
                    "Undefined error (21) at line {}: {} not defined".format(
                        lineno, product_sym))
            if self._def[product_sym].process is not None:
                raise ValueError(
                    "Redefined error (22) at line {}: process of material {} is already defined"
                    .format(lineno, product_sym))
            process_index = len(self._process)
            self._def[product_sym].process = process_index
            self._process.append(
                Process(materials, machine_sym, time, produce_rate, acc_sym))

        else:
            raise ValueError(
                "Syntax error (23) at line {}: unknown command".format(lineno))

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
            if def_.process is None or def_.is_source:
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

        no_product = True
        for sym, def_ in self._def.items():
            if def_.is_product:
                _visitor(sym)
                no_product = False

        if no_product:
            raise ValueError("no product indicated")

    def _bfs_1(self):
        q = deque()  # type: T.Deque[str]
        successor_visited = {}  # type: T.Dict[str, int]

        for sym, def_ in self._def.items():
            if def_.is_product:
                def_.need = def_.product_need
                if (not sym in self._successor) or len(
                        self._successor[sym]) == 0:
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

            max_p_rate = min_with_none(def_.max_rate_from_material,
                                       def_.max_rate_from_machine)

            if not sym in self._successor:
                continue

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
                if (not sym in self._successor) or len(
                        self._successor[sym]) == 0:
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
                if pred_def.max_rate_from_consumer is not None:
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

            label_caption_part = [def_.name]
            label_detail_part = [
                "{:.6g}/s".format(def_.actual_rate),
            ]
            if (not def_.is_source) and def_.process is not None:
                process = self._process[def_.process]
                machine = self._machine[process.machine]
                label_caption_part.append("{}×{}".format(
                    machine.name, def_.machine_count))
            if def_.is_source:
                label_detail_part.append("source")
            else:
                label_detail_part.append(state_str(def_.state))
            if def_.is_product:
                label_detail_part.append("product")
            label = '<{}<BR /><FONT POINT-SIZE="10">{}</FONT>>'.format(
                ", ".join(label_caption_part), ", ".join(label_detail_part))

            parts = ['label = {}'.format(label)]
            if def_.is_source:
                parts.append('shape = "octagon"')
                parts.append('style = "filled"')
                parts.append('fillcolor = "green3"')
                sources.append(sym)
            else:
                if def_.is_product:
                    parts.append('shape = "doubleoctagon"')
                    products.append(sym)
                else:
                    parts.append('shape = "box"')
                if def_.state == 1:
                    parts.append('style = "filled"')
                    parts.append('fillcolor = "lightpink"')
                elif def_.state == 2:
                    parts.append('style = "filled"')
                    parts.append('fillcolor = "lightskyblue"')
                elif def_.state == 3:
                    parts.append('style = "filled"')
                    parts.append('fillcolor = "greenyellow"')

            print('    def_{} [{}]'.format(sym, ", ".join(parts)))

        print(
            '    t [label = "产物", shape = "plaintext", fontcolor = "violetred"]'
        )

        for sym, def_ in self._def.items():
            if not def_.visited:
                continue
            if def_.process is None or def_.is_source:
                continue
            process = self._process[def_.process]

            real_produce_rate = self._calc_real_produce_rate(process)
            go_sum = 0
            for pred_sym, pred_need in process.materials:
                pred_def = self._def[pred_sym]
                z = pred_need / real_produce_rate
                actual_go = def_.actual_rate * z
                pred_def.succ_output_rate += actual_go
                print(
                    '    def_{} -> def_{} [label = "{:.6g}/s", arrowsize = 0.5]'
                    .format(pred_sym, sym, actual_go))
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
            print(
                '    acc_{} [label = <{}<BR /><FONT POINT-SIZE="10">{}, accelerator</FONT>>, shape = "box", style = "filled", fillcolor = "khaki"]'
                .format(acc_sym, r["name"], "{:.6g}/s".format(r["total"])))

        for r in acc_record:
            print(
                '    acc_{} -> def_{} [label = "{:.6g}/s", style = "dotted", color = "darkgoldenrod", fontcolor = "darkgoldenrod", arrowsize = 0.5]'
                .format(r["acc"], r["product"], r["rate"]))

        for sym in products:
            def_ = self._def[sym]
            product_rate = def_.actual_rate - def_.succ_output_rate
            print(
                '    def_{} -> t [label = "{:.6g}/s", style = "dotted", color = "violetred", fontcolor = "violetred", arrowsize = 0.5]'
                .format(sym, product_rate))

        print('    {{ rank = "same"; {} }}'.format(", ".join(
            ["def_{}".format(x) for x in sources])))

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
