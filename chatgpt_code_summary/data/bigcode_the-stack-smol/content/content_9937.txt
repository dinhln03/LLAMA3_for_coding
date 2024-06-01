from flask import render_template, request, redirect, url_for, flash
from datetime import datetime as dt
from app.forms import AdicionarQuarto, VerificarDisponibilidade
from app.models import Rooms, Hotels, User, Reservation, Status
from app import db


def adicionar_quarto(user_id):
    form_reserva = VerificarDisponibilidade()
    user = User.query.filter_by(id=user_id).first()
    form = AdicionarQuarto()

    if user.profile not in ['admin', 'gerente']:
        return '<h1>Erro! Você não pode acessar este conteúdo!</h1>'

    if user.hotel_id is None:
        hoteis = Hotels.query.order_by(Hotels.created_at)
        form.hotel_id.choices = [(hotel.id, hotel.name) for hotel in hoteis if hotel.user_id == user_id]
    else:
        hoteis = Hotels.query.filter_by(id=user.hotel_id).order_by(Hotels.created_at)
        form.hotel_id.choices = [(hotel.id, hotel.name) for hotel in hoteis]

    if request.method == 'POST':
        if form.validate_on_submit():
            room = Rooms.query.filter_by(hotel_id=form.hotel_id.data, number=form.number.data).first()
            if room is None:
                room = Rooms(number=form.number.data,
                             hotel_id=form.hotel_id.data,
                             name=form.name.data,
                             short_description=form.short_description.data,
                             kind=form.kind.data,
                             phone_extension=form.phone_extension.data,
                             price=float(form.price.data.replace('.','').replace(',','.')),
                             guest_limit=form.guest_limit.data)
                db.session.add(room)
                db.session.commit()

                flash('Quarto cadastrado com sucesso!', 'success')
            else:
                flash('Quarto já existe...', 'danger')
        return redirect(url_for('ocupacao_quartos_endpoint', id=form.hotel_id.data))

    return render_template('adicionar_quartos.html',
                           form=form,
                           hoteis=hoteis,
                           user=user,
                           titulo='Adicionar quarto',
                           form_reserva=form_reserva
                           )


def ocupacao_quartos(id, user_id):
    form_reserva = VerificarDisponibilidade()
    user = User.query.filter_by(id=user_id).first()
    hotel = Hotels.query.get_or_404(id)
    if hotel.user_id != user_id and user.hotel_id != hotel.id:
        return '<h1>Erro! Você não pode acessar este conteúdo!</h1>'
    quartos = Rooms.query.filter_by(hotel_id=id).order_by(Rooms.number)
    reservas = Reservation.query.order_by(Reservation.id)
    hoje = dt.strptime(dt.today().strftime('%Y-%m-%d'), '%Y-%m-%d')
    status_reservas = [(r.room_id, (r.check_in <= hoje <= r.check_out)) for r in reservas if r.status == Status.ATIVO]
    status_reservas = [status for status in status_reservas if status[1] is True]
    status_reservas = dict(set(status_reservas))
    return render_template('ocupacao_quartos.html',
                           quartos=quartos,
                           form_reserva=form_reserva,
                           status_reservas=status_reservas
                           )


def deletar_quarto(id_quarto, user_id):
    user = User.query.filter_by(id=user_id).first()
    quarto = Rooms.query.get_or_404(id_quarto)
    id_hotel = quarto.hotel_id
    hotel = Hotels.query.get_or_404(id_hotel)
    if hotel.user_id != user_id and user.hotel_id != hotel.id or user.profile not in ['admin', 'gerente']:
        return '<h1>Erro! Você não pode acessar este conteúdo!</h1>'
    db.session.delete(quarto)
    db.session.commit()
    flash('Quarto deletado com sucesso!', 'success')
    return redirect(f'/ocupacao-quartos/{id_hotel}')


def editar_quarto(quarto_id, user_id):
    form_reserva = VerificarDisponibilidade()
    form = AdicionarQuarto()
    user = User.query.filter_by(id=user_id).first()

    user_id_room = Rooms \
        .query.filter_by(id=quarto_id) \
        .join(Hotels, Rooms.hotel_id == Hotels.id).add_columns(Hotels.user_id).add_columns(Hotels.id)
    if [i.user_id for i in user_id_room][0] != user_id and user.hotel_id != [i for i in user_id_room][0][2]:
        return '<h1>Erro! Você não pode acessar este conteúdo!</h1>'

    if user.hotel_id is None:
        hoteis = Hotels.query.order_by(Hotels.created_at)
        form.hotel_id.choices = [(hotel.id, hotel.name) for hotel in hoteis if hotel.user_id == user_id]
    else:
        hoteis = Hotels.query.filter_by(id=user.hotel_id).order_by(Hotels.created_at)
        form.hotel_id.choices = [(hotel.id, hotel.name) for hotel in hoteis]

    if form.validate_on_submit():
        
        if request.method == 'POST':
            to_update = Rooms.query.get_or_404(quarto_id)
            to_update.hotel_id = request.form['hotel_id']
            to_update.number = request.form['number']
            to_update.name = request.form['name']
            to_update.short_description = request.form['short_description']
            to_update.kind = request.form['kind']
            to_update.phone_extension = request.form['phone_extension']
            to_update.price = float(request.form['price'].replace('.','').replace(',','.'))
            to_update.guest_limit = request.form['guest_limit']
            db.session.commit()
            flash('Quarto editado com sucesso!', 'success')
        return redirect(url_for('ocupacao_quartos_endpoint', id=request.form['hotel_id']))

    room = Rooms.query.filter_by(id=quarto_id).first()

    form.hotel_id.default = room.hotel_id
    form.process()
    form.number.data = room.number
    form.name.data = room.name
    form.short_description.data = room.short_description
    form.kind.data = room.kind
    form.phone_extension.data = room.phone_extension
    form.price.data = str(room.price).replace('.',',')
    form.guest_limit.data = room.guest_limit

    return render_template('adicionar_quartos.html',
                           form=form,
                           user=user,
                           quarto=room,
                           titulo='Editar quarto',
                           form_reserva=form_reserva
                           )
