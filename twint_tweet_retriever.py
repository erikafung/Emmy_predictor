# retrieve tweets about nominees
# run this program twice for each nominee
# once to collect tweets relating to them and the show
# once to collect tweets relating to them and to the Emmy's
import twint

c = twint.Config()

# search with nominee name and 'Emmy'
c.Search = "Tracee Ellis Ross Emmy"
# search with nominee name and nominated show
# c.Search = "Tracee Ellis Ross Emmy"
c.Lang = "en"
# limit date to when nominees were announced
c.Since = "2020-07-29"
# limit date to Emmy awards ceremony
c.Until = "2020-09-21"
c.Pandas = True
c.Store_csv = True
# output tweets to csv
c.Output = "Tracee_emmy_tweets.csv"
# outputs tweets to csv
# c.Output = "Tracee_show_tweets.csv"
c.Hide_output = True
twint.run.Search(c)
