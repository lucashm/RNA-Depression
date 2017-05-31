#!/usr/bin/python

# enable debugging
import cgi
import cgitb
cgitb.enable()
form = cgi.FieldStorage()

from time import gmtime, strftime
showtime = strftime("%Y-%m-%d %H:%M:%S", gmtime())


print "Content-Type: text/plain\r\n\r\n"
print

print "Adicionado ao DB com sucesso!"

import sqlite3
conn = sqlite3.connect('/opt/lampp/htdocs/RNA/RNA.db')

c = conn.cursor()

conn.cursor().execute("INSERT INTO PESSOA VALUES (?,?,?,?)", (form.getvalue("nome"),form.getvalue("questions"), form.getvalue("print_final"), showtime))

conn.commit()
conn.close()
