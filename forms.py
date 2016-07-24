from wtforms import Form, FileField

class FileForm(Form):
    csv_file = FileField('csv_file')