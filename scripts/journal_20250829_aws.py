# %%
import boto3
import requests
from botocore.exceptions import ClientError
import os

# %%
ip = requests.get('https://checkip.amazonaws.com').text.strip()
# %%
ec2 = boto3.resource('ec2')

# %%
security_group = ec2.SecurityGroup(os.environ['AWS_SECURITY_GROUP_ID'])

# %%
try:
    security_group.authorize_ingress(
        DryRun=False,
        IpPermissions=[
            {
                'FromPort': 22,
                'ToPort': 22,
                'IpProtocol': 'tcp',
                'IpRanges': [
                    {
                        'CidrIp': f'{ip}/32',
                    }
                ]
            }
        ]
    )
except ClientError as e:
    if e.response['Error']['Code'] == 'InvalidPermission.Duplicate':
        print('Rule already exists')
    else:
        raise e

# %%
ip_permissions = security_group.ip_permissions

# %%
old_ips = [r['CidrIp'] for r in ip_permissions[0]['IpRanges'] if ip not in r['CidrIp']]

# %%
# Revoke access for all other IP addresses
for old_ip in old_ips:
    security_group.revoke_ingress(
        CidrIp=old_ip,
        FromPort=22,
        ToPort=22,
        IpProtocol='tcp',
    )

print(f'Revoked access for {len(old_ips)} old IP addresses')

