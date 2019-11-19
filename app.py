import json

from flask import Flask, render_template
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from nlp.opinion_extraction import news_parser
from nlp.text_summarization import TextSummarizer
from nlp.sentiment_classification import SentimentClassifier
from config import config

text_summarizer = TextSummarizer()
sentiment_classifier = SentimentClassifier()

bootstrap = Bootstrap()

DEFAULT_EXTRACTION = """小明说，今天天气不错。"""



DEFAULT_SUMMARIZED = """
里皮已经回到了欧洲，但他仍有一肚子话想说。
在接受天空体育采访时，里皮表示——自己选择辞去国足主帅是因为继续合作的条件没有了。
“我在中国度过了将近8年的时间，那是一段快乐的时光。带领国家队重要的不是赢得比赛，而是发展运动，我帮助了中国足球发展。”
“之所以选择辞职是因为继续合作的条件没有了。当我相信不再有热情、欲望和信任这些继续合作的自然条件时，我不喜欢赚太多自己不应得的钱。”
关于未来的计划，里皮表示：“我没有重返俱乐部教练席的渴望，我认为自己永远不会再执教俱乐部了。”
"""

DEFAULT_CLASSIFIED = """
在一阵细细熬煮中，品尝是一件很幸福的一件事情，它可以调剂生活当中所遇到的不愉快的事，让您有好心情面对每一天。
我今天来到的贻贝湾，就是一家让人吃的开心，买的放心的良心店​家。它位于奉贤百联广场五楼的中心地段。
在这里可以放宽心的尽情的吃，因为这的每一份量都很丰富。价美物廉的同时又不失质量保证。
这保证了能让您在吃到之前，能够保持最好的鲜味。 给我留下深刻印象的是这的特色牛排。
当牛肉在我口中，渐渐的消融的时候，我感觉到当中有一点点脂肪的感觉，入口即化的细腻口感，这是黄金牛肉的象征。
我相信只要您尝过一次，您就能知道这确实是太好吃了。
另外，您还可以在这里享受到他们无微不至的服务，更可以大快朵颐的吃，因为当你需要帮助的时候，
这里服务员都会很热心的帮你进行热情服务。 我喜欢贻贝湾，相信如果大家来的话​，就会被这里鲜嫩可口的味道深深吸引，
希望大家不要错过。
"""

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    bootstrap.init_app(app)
    
    return app


app = create_app('production')


class InputForm(FlaskForm):
    content = TextAreaField('输入想要分析的内容', validators=[DataRequired()],
                            render_kw={"rows": 10})
    submit = SubmitField('解析')


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html', e=e), 500


@app.route('/')
def hello_world():
    return render_template('base.html')


@app.route('/extract', methods=['GET', 'POST'])
def extract():
    results = None
    form = InputForm()
    form.content.data = DEFAULT_EXTRACTION
    if form.validate_on_submit():
        content = form.content.data
        results = news_parser(content)
    return render_template('extract.html', form=form, results=results)


@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    results = None
    form = InputForm()
    form.content.data = DEFAULT_SUMMARIZED
    if form.validate_on_submit():
        content = form.content.data
        results = text_summarizer.summarize(content)
    return render_template('summarize.html', form=form, results=results)


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    results = None
    form = InputForm()
    form.content.data = DEFAULT_CLASSIFIED
    if form.validate_on_submit():
        content = form.content.data
        results = sentiment_classifier.predict(content)
        results = json.dumps(results)
    return render_template('classify.html', form=form, results=results)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()