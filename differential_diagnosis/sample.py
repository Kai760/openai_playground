import streamlit as st
from prompts import kanbetu_sikkan
import os

# st.image('logo.png', width=200)
st.title('鑑別疾患予測')
st.text('SOAPの記述から疑われる病名を、URLの専門領域を意識しながら予測します')

# OpenAi-API key
key = st.text_input('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = key


# URL入力欄
url = st.text_input('URL')

# SOAP自由記述欄
soap_help='''
適当なSOAPが見つからない場合、以下のSOAPをコピペしてお使いください

<S>一昨日から風邪を引いたみたい
<O>2024/01/01 体温 36.2度 収縮期血圧 153 mmHg 拡張期血圧 99 mmHg 脈拍 81 bpm 体重 82.2 kg自宅血圧：130-150/90-100【腹部超音波】　
肝臓　肝臓内に腫瘤病変なし、脂肪肝　あり　胆嚢　壁肥厚なし　胆石なし　膵臓　腫大なし　脾臓　副脾なし　肥大あり　腎臓　水腎症なし　腹部大動脈　動脈瘤なし
2023/10/25(水) ＡＳＴ 64 U/L[H] ＡＬＴ 96 U/L[H] ＡＬＰ/IFCC 131 U/L[H] γ-ＧＴ 47 U/L[H] 尿酸 7.6 mg/dL[H] 尿素窒素 20.9 mg/dL[H]
'''
soap = st.text_area('SOAP', help=soap_help)

# 生成モード選択
model_ver = '3.5' if st.radio("生成モード", ('速度重視', '品質重視（30秒ほどかかります）')) == '速度重視' else '4'


# 予測実行ボタン
if st.button('予測実行'):
    with st.spinner('解析中...'):
        # diagnose関数を呼び出し
        result = kanbetu_sikkan(url, soap, model_ver)
        # 結果表示
        st.text('予測結果:')
        st.write(result)
