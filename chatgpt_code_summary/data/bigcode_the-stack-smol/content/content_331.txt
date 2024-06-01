import digi
import digi.on as on


@on.control
def h0(c):
    for k, v in c.items():
        v["status"] = v.get("intent",
                            v.get("status", "undef"))


if __name__ == '__main__':
    digi.run()
