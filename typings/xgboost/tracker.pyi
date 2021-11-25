

"""
This script is a variant of dmlc-core/dmlc_tracker/tracker.py,
which is a specialized version for xgboost tasks.
"""
class ExSocket:
    """
    Extension of socket to handle recv and send of special data
    """
    def __init__(self, sock) -> None:
        ...
    
    def recvall(self, nbytes): # -> bytes:
        ...
    
    def recvint(self): # -> Any:
        ...
    
    def sendint(self, n): # -> None:
        ...
    
    def sendstr(self, s): # -> None:
        ...
    
    def recvstr(self): # -> str:
        ...
    


kMagic = ...
def get_some_ip(host): # -> str:
    ...

def get_host_ip(hostIP=...): # -> str | _RetAddress:
    ...

def get_family(addr): # -> AddressFamily:
    ...

class SlaveEntry:
    def __init__(self, sock, s_addr) -> None:
        ...
    
    def decide_rank(self, job_map): # -> Any | Literal[-1]:
        ...
    
    def assign_rank(self, rank, wait_conn, tree_map, parent_map, ring_map):
        ...
    


class RabitTracker:
    """
    tracker for rabit
    """
    def __init__(self, hostIP, nslave, port=..., port_end=...) -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    @staticmethod
    def get_neighbor(rank, nslave): # -> list[Unknown]:
        ...
    
    def slave_envs(self): # -> dict[str, Unknown | int]:
        """
        get enviroment variables for slaves
        can be passed in as args or envs
        """
        ...
    
    def get_tree(self, nslave): # -> tuple[dict[Unknown, Unknown], dict[Unknown, Unknown]]:
        ...
    
    def find_share_ring(self, tree_map, parent_map, r): # -> list[Unknown]:
        """
        get a ring structure that tends to share nodes with the tree
        return a list starting from r
        """
        ...
    
    def get_ring(self, tree_map, parent_map): # -> dict[Unknown, Unknown]:
        """
        get a ring connection used to recover local data
        """
        ...
    
    def get_link_map(self, nslave): # -> tuple[dict[Unknown, Unknown], dict[Unknown, Unknown], dict[Unknown, Unknown]]:
        """
        get the link map, this is a bit hacky, call for better algorithm
        to place similar nodes together
        """
        ...
    
    def accept_slaves(self, nslave):
        ...
    
    def start(self, nslave): # -> None:
        ...
    
    def join(self): # -> None:
        ...
    
    def alive(self): # -> bool:
        ...
    


