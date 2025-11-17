# -*- coding: utf-8 -*-
r"""
"""
from time import time

from phyem.tools.frozen import Frozen
from phyem.src.config import RANK, MASTER_RANK
from phyem.tools.miscellaneous.php import php
from phyem.tools.miscellaneous.timer import MyTimer


class Ranking(Frozen):
    r"""Use it to test the speed of different machines."""

    def __init__(self, ranking_title, record_file_dir, top_how_much=10, num_ranks=5):
        r""""""
        self._ranking_title = ranking_title
        self._record_file_dir = record_file_dir
        self._top_how_much = top_how_much

        self._my_hostname = ''
        self._t_start = - 1.0
        self._current_records = None
        self._all_hosts = None
        self._num_ranks = num_ranks
        self._freeze()

    def start_ranking(self):
        r""""""
        if RANK == MASTER_RANK:

            with open(self._record_file_dir, 'r') as file:
                records = file.readlines()
            file.close()

            current_records = {}
            all_hosts = []
            for record in records:
                ranking, hostname, used_size, time_cost = record.split(' ')
                ranking = int(ranking)
                used_size = int(used_size)
                time_cost = float(time_cost)
                current_records[ranking] = (hostname, used_size, time_cost)
                all_hosts.append(hostname)
            import socket
            self._my_hostname = socket.gethostname()
            self._t_start = time()
            self._current_records = current_records
            self._all_hosts = all_hosts
        else:
            pass

    def report_ranking(self):
        r""""""
        if RANK == MASTER_RANK:
            from phyem.tools.miscellaneous.timer import MyTimer
            t_cost = time() - self._t_start
            if self._my_hostname in self._all_hosts:
                for r in self._current_records:
                    if self._current_records[r][0] == self._my_hostname:
                        self._current_records[r] = (self._my_hostname, self._num_ranks, t_cost)
                        break
                    else:
                        pass
            else:
                num_hosts = len(self._all_hosts)
                self._current_records[num_hosts] = (self._my_hostname, self._num_ranks, t_cost)

            new_ranking = []
            new_cost = []
            new_size = []
            for r in self._current_records:
                hostname, the_size, the_cost = self._current_records[r]
                if not new_ranking:
                    new_ranking.append(hostname)
                    new_cost.append(the_cost)
                    new_size.append(the_size)
                else:
                    insert_i = None
                    for i, _cost in enumerate(new_cost):
                        if the_cost <= _cost:
                            insert_i = i
                            break
                        else:
                            pass
                    if insert_i is None:
                        new_ranking.append(hostname)
                        new_cost.append(the_cost)
                        new_size.append(the_size)
                    else:
                        new_cost.insert(insert_i, the_cost)
                        new_ranking.insert(insert_i, hostname)
                        new_size.insert(insert_i, the_size)

            new_record = {}
            for r, hostname in enumerate(new_ranking):
                if r < self._top_how_much:
                    new_record[r] = (hostname, new_size[r], new_cost[r])
                else:
                    pass

            php(
                f"\n\n----------- RANKING [{self._ranking_title}] SUMMARY ------------",
                flush=True
            )
            my_rank = new_ranking.index(self._my_hostname)
            if my_rank >= self._top_how_much:
                php(f"    [rank*] <{self._my_hostname}> costs {MyTimer.seconds2hmsm(t_cost)}\n", flush=True)
            else:
                php(
                    f"    [rank #{my_rank + 1}] <{self._my_hostname}> costs "
                    f"{MyTimer.seconds2hmsm(t_cost)}\n",
                    flush=True
                )

            php(f'         ***** Ranking (TOP {self._top_how_much}) *****')
            for r in new_record:
                _hostname, _size, _cost = new_record[r]
                php(
                    f"{r + 1}: <{_hostname}> uses {_size} ranks, costs {MyTimer.seconds2hmsm(_cost)}.",
                    flush=True
                )
            php()

            # noinspection PyUnboundLocalVariable
            with open(self._record_file_dir, 'w') as file:
                w_str = []
                for r in new_record:
                    _hostname, _size, _cost = new_record[r]
                    w_str.append(f"{r} {_hostname} {_size} {_cost}")
                w_str = '\n'.join(w_str)
                file.write(w_str)
            file.close()

    def print_ranking(self):
        r""""""
        php(
            f"\n\n----------------- RANKING [{self._ranking_title}] SUMMARY -----------------",
            flush=True
        )

        with open(self._record_file_dir, 'r') as file:
            records = file.readlines()
        file.close()
        for r, line in enumerate(records):
            # print(line.split(' '))
            _, _hostname, _size, _cost = line[:-1].split(' ')
            _cost = float(_cost)
            php(
                f"{r + 1: >3}", f"{_hostname: >15}", f" {_size: >3} ranks : ",
                f"costs {MyTimer.seconds2hmsm(_cost): >15}"
            )
        php(f"==================== END [{self._ranking_title}] RANKING ==================\n\n")
