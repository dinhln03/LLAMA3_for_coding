from flask import (
    Blueprint,
    render_template,
    g,
    Response,
    request,
    redirect,
    url_for,
    abort,
)
from flask.helpers import flash
from flask_login import login_required
from saws.blueprints.utils.utils_ec2 import (
    get_ec2_info,
    get_key_pairs,
    download_key_pair,
    launch_instance,
    stop_instance,
    terminate_instance,
    describe_instace,
    create_tags,
    EC2Instance,
)
from saws.blueprints.utils.utils_lambda import get_lambda_info
from saws.forms import CreateInstanceForm


bp = Blueprint('compute', __name__, url_prefix='/compute')

EC2_STATE_MAP = {
    'pending': 'secondary',
    'running': 'success',
    'shutting-down': 'warning',
    'terminated': 'danger',
    'stopping': 'warning',
    'stopped': 'dark'
}


@bp.route('/ec2', methods=['GET'])
@login_required
def ec2():
    instances = get_ec2_info(g.user.account)
    return render_template('compute/ec2.html', ins=instances, state_map=EC2_STATE_MAP)


@bp.route('/ec2/<id>', methods=['GET'])
@login_required
def instance(id):
    instance = describe_instace(g.user.account, id)
    if not instance:
        abort(404, 'instance not found')
    instance_object = EC2Instance(instance)
    return render_template('compute/ec2_instance.html', i=instance_object, state_map=EC2_STATE_MAP)


@bp.route('/ec2/<id>/name', methods=['POST'])
@login_required
def instance_name(id):
    name = request.form.get('instance_name')
    tags = [{'Key': 'Name', 'Value': name}]
    create_tags(g.user.account, id, tags)

    flash(f'Name changed to {name}', 'success')

    return redirect(url_for('compute.instance', id=id))


@bp.route('/ec2/create', methods=['GET', 'POST'])
@login_required
def instance_create():
    instance_form = CreateInstanceForm(request.form)
    print(request.method)
    if request.method == 'POST':
        if instance_form.validate():
            os = request.form.get('os')
            size = request.form.get('size')
            key_name = request.form.get('key_pair')
            port_22 = request.form.get('port_22')
            port_80 = request.form.get('port_80')

            print(f'Launching {os} {size} with {port_22} {port_80}')

            props = {
                'key_name':key_name,
            }
            # TODO: create sg
            launch_instance(g.user.account, props)
            flash('Launching instance', 'success')
            return redirect(url_for('compute.ec2'))

    keys = get_key_pairs(g.user.account)
    return render_template('compute/ec2_create.html', keys=keys, form=instance_form)


@bp.route('/ec2/stop/<instance>', methods=['GET'])
@login_required
def instance_stop(instance):
    if not instance:
        abort(400)

    stop_instance(g.user.account, instance)
    flash(f'Stopping instance {instance}', 'success')

    return redirect(url_for('compute.ec2'))


@bp.route('/ec2/terminate/<instance>', methods=['GET'])
@login_required
def instance_terminate(instance):
    if not instance:
        abort(400)

    terminate_instance(g.user.account, instance)
    flash(f'Terminating instance {instance}', 'success')

    return redirect(url_for('compute.ec2'))


@bp.route('/ec2/keypair', methods=['GET'])
@login_required
def keypair():
    keys = get_key_pairs(g.user.account)
    return render_template('compute/ec2_keypair.html', keys=keys)


@bp.route('/ec2/keypair/download/<name>', methods=['GET'])
@login_required
def download_keypair(name):
    kp = download_key_pair(g.user.account, name)
    return Response(kp['KeyMaterial'], mimetype='application/x-binary')


@bp.route('/functions', methods=['GET'])
@login_required
def functions():
    lambdas = get_lambda_info(g.user.account)
    return render_template('compute/lambda.html', lambdas=lambdas)


@bp.route('/functions/<id>', methods=['GET'])
@login_required
def single_function(id):
    return render_template('compute/lambda.html')