from apps.index_qt.models import Member, MemberType


def member_verification(user_id):
    # 查询该用户的用户等级
    grade = Member.objects.get(user_id=user_id)
    print(grade.now_number)
    # 查询当前用户的等级
    member = MemberType.objects.get(id=grade.member_type_id)
    if int(grade.now_number) >member.number:
        return False
    return True