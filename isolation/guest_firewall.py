def policy():
    return """#!/usr/sbin/nft -f
flush ruleset
table inet supportagent {
  chain input {
    type filter hook input priority 0; policy drop;
    iifname "lo" accept
    ct state established,related accept
  }
  chain output {
    type filter hook output priority 0; policy drop;
    oifname "lo" accept
    ct state established,related accept
  }
}
"""


if __name__ == "__main__":
    print(policy())
