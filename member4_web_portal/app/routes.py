from flask import Blueprint, render_template

main = Blueprint("main", __name__)


@main.route("/")
def index():
    return render_template("index.html")


@main.route("/heart")
def heart():
    return render_template("heart.html")


@main.route("/xray")
def xray():
    return render_template("xray.html")


@main.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")
