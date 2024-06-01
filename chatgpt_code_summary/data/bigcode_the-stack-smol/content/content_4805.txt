#!/usr/bin/env python3

import json
import argparse
import datetime


class TicketManager:
    ticketfile = '/Users/ben/ticketing/tickets.json'

    def __init__(self: object, ticketfile: str='/Users/ben/Google Drive/code/ticketing/tickets.json')->object:
        self.ticketfille = ticketfile
        self.read_tickets()

    def read_tickets(self)-> None:
        self.tickets = json.load(open(self.ticketfile))

    def write_tickets(self)-> None:
        json.dump(self.tickets, open(self.ticketfile, "w"), indent=4)

    def create_ticket(self, title="", desc="", dest="", due="", pri=0, completed=False):
        ticket = {"title": title,
                  "desc": desc,
                  "for": dest,
                  "time_in": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                  "time_out": due,
                  "nice": pri,
                  "completed": completed
                  }
        self.tickets[title] = ticket
        self.write_tickets()
        self.read_tickets()

    def update_ticket(self, title, new_completed):
        self.tickets[title]["completed"] = new_completed
        self.write_tickets()
        self.read_tickets()

    def show_all_tickets(self):
        for ticket in self.tickets.values():
            print("""TICKET NAME: {}
\tTICKET DESCRIPTION: {}
\tTICKET CREATED: {}
\tTICKET DUE: {}
\tTICKET FOR: {}
\tTICKET DONE: {}
\tTICKET PRIORITY: {}
            """.format(ticket['title'], ticket['desc'], ticket['time_in'], ticket['time_out'],
                       ticket['for'], ticket['completed'], ticket['nice']))

    def show_unifnished(self):
        flag = False
        for ticket in self.tickets.values():
            if not ticket['completed']:
                flag = True
                print("""TICKET NAME: {}
\tTICKET DESCRIPTION: {}
\tTICKET CREATED: {}
\tTICKET DUE: {}
\tTICKET FOR: {}
\tTICKET PRIORITY: {}
        """.format(ticket['title'], ticket['desc'], ticket['time_in'], ticket['time_out'],
                   ticket['for'], ticket['nice']))
        if not flag:
            print("No Unfinished Tasks!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--mode", action="store", dest="mode", default='ls')
    parser.add_argument("--title", action="store", dest="title")
    parser.add_argument("--desc", action="store", dest="desc")
    parser.add_argument("--for", action="store", dest="dest")
    parser.add_argument("--due", action="store", dest="time_out")
    parser.add_argument("--pri", action="store", dest="nice")
    parser.add_argument("--done", action="store_true",
                        dest="completed", default=False)
    args = parser.parse_args()

    tm = TicketManager("tickets.json")

    if args.mode == "ls":
        tm.show_unifnished()
    elif args.mode == "ls2":
        tm.show_all_tickets()
    elif args.mode == "new" or args.mode == "add":
        tm.create_ticket(title=args.title, desc=args.desc, dest=args.dest,
                         due=args.time_out, pri=args.nice, completed=args.completed)
        print("New Task '{}'  Added".format(args.title))
    elif args.mode == "up":
        tm.update_ticket(args.title, args.completed)
