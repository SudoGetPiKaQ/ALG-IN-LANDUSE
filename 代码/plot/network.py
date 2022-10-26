import matplotlib.pyplot as plt
import networkx as nx
import src



my_chinese_dict=dict(fontsize=12,
              color='black',
              family='Heiti TC',
              weight='light',
              )

fig=plt.figure (figsize=(12,5) ,dpi=300)

fig.tight_layout()#调整整体空白


G = nx.Graph()

index_ls = ['商场点密度', '高程', '到道路距离', '到水体距离', '到地铁站距离', '到高速出入口距离', '到铁路距离', '坡度', '到区中心距离', '到机场距离', '学校点密度',
            '夜灯强度', '景点点密度', '到火车站距离', '医院点密度','用地类别']

[G.add_weighted_edges_from([(e[0][0], e[0][1], e[1])]) for e in src.hc_draw  ]


# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

pos = nx.circular_layout(G)  # positions for all nodes

plt.subplot(121)


plt.subplots_adjust(wspace =0.1, hspace =0)

classEdge=[[e[0][0], e[0][1]] for e in src.hc_draw if e[0][0]=='用地类别' or e[0][1]=='用地类别']

lightEdge=[[e[0][0], e[0][1]] for e in src.hc_draw if (e[0][0]=='夜灯强度' or e[0][1]=='夜灯强度') and e[0][1]!='用地类别']

otherEdge=[[e[0][0], e[0][1]] for e in src.hc_draw if e[0][0]!='用地类别' and e[0][1]!='用地类别' and e[0][0]!='夜灯强度' and e[0][1]!='夜灯强度']



plt.xlabel('HC结构',fontdict=my_chinese_dict)

nx.draw_networkx_nodes(G, pos, node_size=1000,node_color='black',alpha=0.2)

# labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family="Heiti TC",font_weight='light')

# edges
# nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
# nx.draw_networkx_edges(
#     G, pos, edgelist=esmall, width=6, alpha=0.5, style="dashed"
# )
nx.draw_networkx_edges(
    G, pos, width=1, alpha=0.5,style="dashed",edgelist=lightEdge
)

nx.draw_networkx_edges(
    G, pos, width=1, alpha=1,edgelist=otherEdge
)


nx.draw_networkx_edges(
    G, pos, width=0.5, alpha=0.2,style="dashed",edgelist=classEdge
)

G2 = nx.Graph()

[G2.add_weighted_edges_from([(e[0][0], e[0][1], e[1])]) for e in src.tan_draw]
# if e[0][0] != 'class' and e[0][1] != 'class'


# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

pos = nx.circular_layout(G2)  # positions for all nodes

plt.subplot(122)
plt.xlabel('TAN结构',fontdict=my_chinese_dict)

nx.draw_networkx_nodes(G2, pos, node_size=1000,node_color='black',alpha=0.2)

# labels
nx.draw_networkx_labels(G2, pos, font_size=10, font_family="Heiti TC",font_weight='light')

classEdge=[[e[0][0], e[0][1]] for e in src.tan_draw if e[0][0]=='用地类别' or e[0][1]=='用地类别']

lightEdge=[[e[0][0], e[0][1]] for e in src.tan_draw if e[0][0]=='夜灯强度' or e[0][1]=='夜灯强度' and e[0][0]!='用地类别']

otherEdge=[[e[0][0], e[0][1]] for e in src.tan_draw if e[0][0]!='用地类别' and e[0][1]!='用地类别' and e[0][0]!='夜灯强度' and e[0][1]!='夜灯强度']

nx.draw_networkx_edges(
    G2, pos, width=1, alpha=0.5,style="dashed",edgelist=lightEdge
)

nx.draw_networkx_edges(
    G2, pos, width=1, alpha=1,edgelist=otherEdge
)


nx.draw_networkx_edges(
    G2, pos, width=0.5, alpha=0.2,style="dashed",edgelist=classEdge
)
# plt.show()
plt.savefig("./structure.png",dpi=300)
