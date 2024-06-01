from util.dc_verilog_parser import *

def main():
    folder = "../dc/sub/adder8/"
    # folder = "../dc/boom/implementation/"
    total_nodes = 0
    total_edges = 0
    ntype = set()
    for v in os.listdir(folder):
        if v.startswith("hier"):
            continue
        vf = os.path.join(folder, v)
        print("parsing {}...".format(vf))
        # parser = DcParser("BoomCore", ["alu_DP_OP", "add_x"])
        parser = DcParser("test1", [ "add_x"], "hadd_s")
        nodes, edges = parser.parse(vf, label_region=True)
        print("nodes {}, edges {}".format(len(nodes), len(edges)))
        # nodes, edges = parser.clip(nodes, edges)
        nodes, edges = parser.remove_div(nodes, edges)
        adder_out_type = collections.defaultdict(int)
        node_type = collections.defaultdict(int)
        for n in nodes:
            if n[1]["is_output"]:
                adder_out_type[n[1]["type"]] += 1
            node_type[n[1]["type"]] += 1
        print(node_type)
        print(adder_out_type)
        print("clipped: nodes {}, edges {}".format(len(nodes), len(edges)))

        for n in nodes:
            ntype.add(n[1]["type"])
        total_nodes += len(nodes)
        total_edges += len(edges)
        # return
    print(ntype)
    print(total_nodes, total_edges)


if __name__ == "__main__":
    # dc_parser("../dc/simple_alu/implementation/alu_d0.20_r2_bounded_fanout_adder.v")
    main()
    # cProfile.run("main()")