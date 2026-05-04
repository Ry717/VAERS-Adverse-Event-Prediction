from sentence_transformers import SentenceTransformer, util

# 測試兩句醫療診斷
sentences = [
    "Patient has Myalgia.", 
    "Patient feels muscle pain."
]

print("=======================================")
print("正在載入【通用模型】 all-MiniLM-L6-v2 ...")
model_general = SentenceTransformer('all-MiniLM-L6-v2')
emb_gen = model_general.encode(sentences)
sim_gen = util.cos_sim(emb_gen[0], emb_gen[1])[0][0].item()

print("\n正在載入【醫療模型】 S-PubMedBert-MS-MARCO ...")
model_medical = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
emb_med = model_medical.encode(sentences)
sim_med = util.cos_sim(emb_med[0], emb_med[1])[0][0].item()

print("=======================================")
print(f"➤ 通用模型認為這兩句話的相似度： {sim_gen:.2%} ")
print(f"➤ 醫療模型認為這兩句話的相似度： {sim_med:.2%} ")
print("=======================================")