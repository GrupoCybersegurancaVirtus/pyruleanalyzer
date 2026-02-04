def predict(sample):
    if sample.get('dur', 0) <= 3.650000735433423e-06:
        if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
            if sample.get('sbytes', 0) <= 5.78165554543375e-06:
                if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                    if sample.get('dur', 0) <= 0.007677176501601934:
                        if sample.get('rate', 0) <= 7.378285545200924e-06:
                            if sample.get('spkts', 0) <= 4.6970410039648414e-05:
                                if sample.get('ct_src_ltm', 0) <= 0.0677966084331274:
                                    if sample.get('sbytes', 0) <= 7.662434882149682e-07:
                                        if sample.get('dur', 0) <= 0.0046772342175245285:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        if sample.get('smean', 0) <= 0.012162161991000175:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                        if sample.get('sjit', 0) <= 0.0038980711833573878:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                            return 2
                                        else:
                                            return 3
                            else:
                                if sample.get('smean', 0) <= 0.9533783793449402:
                                    if sample.get('sbytes', 0) <= 6.19960633230221e-06:
                                        return 2
                                    else:
                                        if sample.get('spkts', 0) <= 0.00046970410039648414:
                                            return 5
                                        else:
                                            return 3
                                else:
                                    return 4
                        else:
                            if sample.get('sbytes', 0) <= 0.0001496264540037373:
                                if sample.get('ct_dst_sport_ltm', 0) <= 0.10810810327529907:
                                    if sample.get('sbytes', 0) <= 4.458144076124881e-06:
                                        if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        return 5
                                else:
                                    if sample.get('ct_srv_src', 0) <= 0.15322580188512802:
                                        if sample.get('ct_srv_src', 0) <= 0.17741934955120087:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('sbytes', 0) <= 4.1446807017564424e-06:
                                            return 6
                                        else:
                                            return 4
                            else:
                                if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                    if sample.get('dbytes', 0) <= 1.6237387171713635e-05:
                                        return 3
                                    else:
                                        if sample.get('smean', 0) <= 0.09864864870905876:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('dur', 0) <= 0.3091546297073364:
                                        if sample.get('ct_src_dport_ltm', 0) <= 0.22413793206214905:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        return 4
                    else:
                        if sample.get('sload', 0) <= 0.0057913989294320345:
                            if sample.get('rate', 0) <= 2.3187964757198642e-06:
                                if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                    if sample.get('spkts', 0) <= 0.0003287928702775389:
                                        if sample.get('smean', 0) <= 0.007432432379573584:
                                            return 6
                                        else:
                                            return 5
                                    else:
                                        if sample.get('sbytes', 0) <= 0.00016230430992436595:
                                            return 4
                                        else:
                                            return 2
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.6065573692321777:
                                        if sample.get('sbytes', 0) <= 5.78165554543375e-06:
                                            return 4
                                        else:
                                            return 3
                                    else:
                                        return 7
                            else:
                                if sample.get('swin', 0) <= 0.5:
                                    if sample.get('sjit', 0) <= 2.805455278576119e-05:
                                        if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sload', 0) <= 8.186965715140104e-06:
                                            return 7
                                        else:
                                            return 5
                                else:
                                    if sample.get('sbytes', 0) <= 0.0003225188556825742:
                                        if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('service', 0) <= 0.0833333358168602:
                                            return 5
                                        else:
                                            return 4
                        else:
                            if sample.get('proto', 0) <= 0.803846150636673:
                                if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                    if sample.get('sbytes', 0) <= 4.94575351694948e-06:
                                        if sample.get('proto', 0) <= 0.5230769217014313:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        if sample.get('rate', 0) <= 0.1180555522441864:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('smean', 0) <= 0.012162161991000175:
                                        if sample.get('proto', 0) <= 0.5230769217014313:
                                            return 6
                                        else:
                                            return 2
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.28688524663448334:
                                            return 3
                                        else:
                                            return 6
                            else:
                                if sample.get('ct_srv_src', 0) <= 0.10483870655298233:
                                    if sample.get('sload', 0) <= 0.00813085213303566:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.6311475336551666:
                                            return 4
                                        else:
                                            return 6
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.5327868759632111:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.2983870953321457:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.3870967850089073:
                                            return 4
                                        else:
                                            return 6
                else:
                    if sample.get('proto', 0) <= 0.803846150636673:
                        if sample.get('spkts', 0) <= 0.008736495859920979:
                            if sample.get('sbytes', 0) <= 0.0011724918731488287:
                                if sample.get('proto', 0) <= 0.857692301273346:
                                    if sample.get('spkts', 0) <= 0.00046970410039648414:
                                        return 1
                                    else:
                                        if sample.get('smean', 0) <= 0.23040539771318436:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('sjit', 0) <= 0.0005328919214662164:
                                        if sample.get('spkts', 0) <= 0.001972757192561403:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('service', 0) <= 0.0833333358168602:
                                            return 6
                                        else:
                                            return 3
                            else:
                                if sample.get('spkts', 0) <= 0.007891028420999646:
                                    if sample.get('service', 0) <= 0.0833333358168602:
                                        if sample.get('sbytes', 0) <= 0.0007461818458978087:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        return 3
                                else:
                                    return 3
                        else:
                            if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                    if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                        return 6
                                    else:
                                        if sample.get('sload', 0) <= 0.07323231920599937:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('smean', 0) <= 0.1618243232369423:
                                        if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                            return 6
                                        else:
                                            return 7
                                    else:
                                        if sample.get('sload', 0) <= 0.8675018846988678:
                                            return 4
                                        else:
                                            return 3
                            else:
                                if sample.get('ct_dst_ltm', 0) <= 0.12068965286016464:
                                    if sample.get('sload', 0) <= 0.15132877230644226:
                                        if sample.get('proto', 0) <= 0.8192307651042938:
                                            return 2
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_src_ltm', 0) <= 0.06779661029577255:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    return 6
                    else:
                        if sample.get('smean', 0) <= 0.016554053872823715:
                            if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                                if sample.get('ct_src_ltm', 0) <= 0.09322033822536469:
                                    if sample.get('state', 0) <= 0.75:
                                        if sample.get('ct_srv_src', 0) <= 0.10483870655298233:
                                            return 2
                                        else:
                                            return 4
                                    else:
                                        return 3
                                else:
                                    if sample.get('proto', 0) <= 0.8961538374423981:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        return 6
                            else:
                                if sample.get('sload', 0) <= 0.08754745125770569:
                                    if sample.get('smean', 0) <= 0.02567567490041256:
                                        if sample.get('sbytes', 0) <= 5.920972398598678e-06:
                                            return 6
                                        else:
                                            return 5
                                    else:
                                        if sample.get('sbytes', 0) <= 1.239921266460442e-05:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('rate', 0) <= 0.75:
                                        if sample.get('sbytes', 0) <= 5.8164845540886745e-05:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.09836065396666527:
                                            return 6
                                        else:
                                            return 4
                        else:
                            if sample.get('ct_srv_src', 0) <= 0.13709677010774612:
                                if sample.get('spkts', 0) <= 0.0007515265315305442:
                                    if sample.get('smean', 0) <= 0.041554054245352745:
                                        return 3
                                    else:
                                        return 6
                                else:
                                    if sample.get('service', 0) <= 0.2083333283662796:
                                        return 3
                                    else:
                                        return 3
                            else:
                                if sample.get('dmean', 0) <= 0.028999999165534973:
                                    if sample.get('djit', 0) <= 3.7096705455041956e-05:
                                        return 3
                                    else:
                                        return 7
                                else:
                                    if sample.get('dur', 0) <= 0.0031455005519092083:
                                        if sample.get('ackdat', 0) <= 0.00471408199518919:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('spkts', 0) <= 0.016063880175352097:
                                            return 3
                                        else:
                                            return 3
            else:
                if sample.get('ct_srv_dst', 0) <= 0.12295081466436386:
                    if sample.get('dur', 0) <= 0.51393723487854:
                        if sample.get('ct_src_ltm', 0) <= 0.14406779408454895:
                            if sample.get('proto', 0) <= 0.8269230723381042:
                                if sample.get('smean', 0) <= 0.013513513375073671:
                                    if sample.get('smean', 0) <= 0.013513513375073671:
                                        return 4
                                    else:
                                        return 6
                                else:
                                    if sample.get('rate', 0) <= 0.0912698358297348:
                                        if sample.get('ct_src_ltm', 0) <= 0.050847455859184265:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sbytes', 0) <= 4.667119469559111e-06:
                                            return 3
                                        else:
                                            return 2
                            else:
                                if sample.get('sload', 0) <= 0.008178760763257742:
                                    if sample.get('ct_dst_ltm', 0) <= 0.07758620753884315:
                                        if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('rate', 0) <= 0.09545454755425453:
                                            return 4
                                        else:
                                            return 1
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                        return 3
                                    else:
                                        if sample.get('sbytes', 0) <= 1.156331063612015e-05:
                                            return 0
                                        else:
                                            return 4
                        else:
                            if sample.get('dbytes', 0) <= 5.635328489006497e-05:
                                if sample.get('ct_state_ttl', 0) <= 0.0833333358168602:
                                    return 6
                                else:
                                    if sample.get('dttl', 0) <= 0.11857707798480988:
                                        if sample.get('ct_src_ltm', 0) <= 0.12711864709854126:
                                            return 2
                                        else:
                                            return 5
                                    else:
                                        if sample.get('response_body_len', 0) <= 1.8215179807157256e-05:
                                            return 3
                                        else:
                                            return 3
                            else:
                                if sample.get('ct_state_ttl', 0) <= 0.1666666716337204:
                                    return 3
                                else:
                                    if sample.get('sload', 0) <= 0.04290052875876427:
                                        if sample.get('ct_src_ltm', 0) <= 0.20338983088731766:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        return 3
                    else:
                        if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                            if sample.get('sbytes', 0) <= 1.0100482086272677e-05:
                                if sample.get('ct_srv_dst', 0) <= 0.07377048954367638:
                                    if sample.get('ct_state_ttl', 0) <= 0.1666666716337204:
                                        if sample.get('proto', 0) <= 0.8153845965862274:
                                            return 2
                                        else:
                                            return 5
                                    else:
                                        return 4
                                else:
                                    if sample.get('ct_srv_src', 0) <= 0.04838709533214569:
                                        if sample.get('ct_srv_src', 0) <= 0.032258063554763794:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('rate', 0) <= 1.3780005701846676e-05:
                                            return 2
                                        else:
                                            return 3
                            else:
                                if sample.get('proto', 0) <= 0.8846153616905212:
                                    if sample.get('ct_src_dport_ltm', 0) <= 0.017241379246115685:
                                        return 7
                                    else:
                                        if sample.get('proto', 0) <= 0.8346153795719147:
                                            return 5
                                        else:
                                            return 1
                                else:
                                    if sample.get('sbytes', 0) <= 2.479842532920884e-05:
                                        if sample.get('proto', 0) <= 0.9038461446762085:
                                            return 6
                                        else:
                                            return 2
                                    else:
                                        if sample.get('rate', 0) <= 0.08712121099233627:
                                            return 4
                                        else:
                                            return 6
                        else:
                            if sample.get('sbytes', 0) <= 5.363704531191615e-06:
                                if sample.get('ct_dst_src_ltm', 0) <= 0.06451612710952759:
                                    if sample.get('stcpb', 0) <= 0.03001263178884983:
                                        if sample.get('sinpkt', 0) <= 0.007485614391043782:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 4
                                        else:
                                            return 6
                                else:
                                    if sample.get('ct_srv_src', 0) <= 0.11290322244167328:
                                        if sample.get('sjit', 0) <= 0.0013408281520241871:
                                            return 8
                                        else:
                                            return 3
                                    else:
                                        if sample.get('smean', 0) <= 0.11216216161847115:
                                            return 2
                                        else:
                                            return 6
                            else:
                                if sample.get('proto', 0) <= 0.8269230723381042:
                                    if sample.get('sload', 0) <= 0.00932253384962678:
                                        if sample.get('sbytes', 0) <= 9.027741543832235e-05:
                                            return 2
                                        else:
                                            return 5
                                    else:
                                        if sample.get('sload', 0) <= 0.00348429494979996:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                        if sample.get('dur', 0) <= 2.750000405171704e-07:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('service', 0) <= 0.0833333358168602:
                                            return 6
                                        else:
                                            return 5
                else:
                    if sample.get('proto', 0) <= 0.8269230723381042:
                        if sample.get('dur', 0) <= 0.9984217584133148:
                            if sample.get('rate', 0) <= 1.6932809785430436e-06:
                                if sample.get('sbytes', 0) <= 3.935705194635375e-06:
                                    return 4
                                else:
                                    if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                        if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('rate', 0) <= 8.569443343731109e-05:
                                            return 5
                                        else:
                                            return 3
                            else:
                                if sample.get('sjit', 0) <= 0.00015147735393838957:
                                    if sample.get('spkts', 0) <= 0.001690934703219682:
                                        if sample.get('ct_src_ltm', 0) <= 0.016949152573943138:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sjit', 0) <= 3.410621684452053e-05:
                                            return 4
                                        else:
                                            return 0
                                else:
                                    if sample.get('proto', 0) <= 0.857692301273346:
                                        if sample.get('proto', 0) <= 0.696153849363327:
                                            return 3
                                        else:
                                            return 0
                                    else:
                                        if sample.get('dinpkt', 0) <= 6.7015257485536495e-06:
                                            return 3
                                        else:
                                            return 4
                        else:
                            if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                                if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                    if sample.get('ct_src_ltm', 0) <= 0.19491524994373322:
                                        if sample.get('sttl', 0) <= 0.9980392158031464:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 5
                                        else:
                                            return 3
                                else:
                                    if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                        if sample.get('ct_srv_dst', 0) <= 0.180327869951725:
                                            return 2
                                        else:
                                            return 0
                                    else:
                                        if sample.get('sbytes', 0) <= 2.243003700641566e-05:
                                            return 3
                                        else:
                                            return 2
                            else:
                                if sample.get('sinpkt', 0) <= 9.035276941204984e-05:
                                    if sample.get('dur', 0) <= 0.0007521584630012512:
                                        return 5
                                    else:
                                        return 3
                                else:
                                    if sample.get('proto', 0) <= 0.8769230544567108:
                                        return 6
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.25:
                                            return 4
                                        else:
                                            return 6
                    else:
                        if sample.get('sbytes', 0) <= 6.0602892517636064e-06:
                            if sample.get('sload', 0) <= 0.008335441816598177:
                                if sample.get('sload', 0) <= 3.0336828871213584e-07:
                                    if sample.get('sbytes', 0) <= 1.567316212458536e-05:
                                        if sample.get('sinpkt', 0) <= 3.749375565575974e-07:
                                            return 4
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sbytes', 0) <= 2.479842532920884e-05:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('rate', 0) <= 0.029040402732789516:
                                        if sample.get('smean', 0) <= 0.031756757758557796:
                                            return 3
                                        else:
                                            return 4
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 2
                                        else:
                                            return 3
                            else:
                                if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                    if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                        return 7
                                    else:
                                        if sample.get('dur', 0) <= 0.6971467733383179:
                                            return 6
                                        else:
                                            return 6
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.6311475336551666:
                                        return 4
                                    else:
                                        return 6
                        else:
                            if sample.get('ct_dst_src_ltm', 0) <= 0.16129031777381897:
                                if sample.get('sload', 0) <= 0.014495754148811102:
                                    if sample.get('sbytes', 0) <= 1.156331063612015e-05:
                                        return 4
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.17213114351034164:
                                            return 2
                                        else:
                                            return 4
                                else:
                                    if sample.get('sload', 0) <= 0.018440176732838154:
                                        return 3
                                    else:
                                        return 1
                            else:
                                if sample.get('proto', 0) <= 0.45384617149829865:
                                    if sample.get('sbytes', 0) <= 1.156331063612015e-05:
                                        if sample.get('rate', 0) <= 0.10555555298924446:
                                            return 3
                                        else:
                                            return 7
                                    else:
                                        if sample.get('sinpkt', 0) <= 2.749542105107139e-07:
                                            return 2
                                        else:
                                            return 3
                                else:
                                    if sample.get('sjit', 0) <= 0.012400660663843155:
                                        if sample.get('sbytes', 0) <= 9.264580285162083e-06:
                                            return 5
                                        else:
                                            return 3
                                    else:
                                        return 4
        else:
            if sample.get('service', 0) <= 0.0416666679084301:
                if sample.get('ct_dst_src_ltm', 0) <= 0.15322580188512802:
                    if sample.get('smean', 0) <= 0.02162162121385336:
                        if sample.get('ct_state_ttl', 0) <= 0.6666666716337204:
                            if sample.get('sbytes', 0) <= 3.329676223984279e-05:
                                if sample.get('ct_dst_ltm', 0) <= 0.06034482829272747:
                                    if sample.get('dbytes', 0) <= 9.64691789704375e-05:
                                        if sample.get('sttl', 0) <= 0.6196078434586525:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dloss', 0) <= 0.0002723806173889898:
                                            return 3
                                        else:
                                            return 7
                                else:
                                    if sample.get('dinpkt', 0) <= 0.001359125948511064:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.05645161122083664:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('djit', 0) <= 0.0020804255618713796:
                                            return 3
                                        else:
                                            return 6
                            else:
                                if sample.get('sload', 0) <= 1.4235353091862635e-06:
                                    if sample.get('rate', 0) <= 3.684280818561092e-05:
                                        return 3
                                    else:
                                        return 2
                                else:
                                    if sample.get('dbytes', 0) <= 1.207570335282071e-05:
                                        if sample.get('dmean', 0) <= 0.04166666604578495:
                                            return 6
                                        else:
                                            return 2
                                    else:
                                        if sample.get('dloss', 0) <= 0.0009987288794945925:
                                            return 6
                                        else:
                                            return 3
                        else:
                            if sample.get('dload', 0) <= 0.0001930992293637246:
                                if sample.get('dbytes', 0) <= 2.1217761968728155e-05:
                                    if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                        if sample.get('smean', 0) <= 0.02060810849070549:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('tcprtt', 0) <= 0.02476105373352766:
                                            return 3
                                        else:
                                            return 3
                                else:
                                    if sample.get('sbytes', 0) <= 3.824251689366065e-05:
                                        if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                            return 7
                                        else:
                                            return 4
                                    else:
                                        if sample.get('sbytes', 0) <= 5.9209723985986784e-05:
                                            return 3
                                        else:
                                            return 3
                            else:
                                if sample.get('sloss', 0) <= 0.0004700131685240194:
                                    if sample.get('smean', 0) <= 0.042905405163764954:
                                        if sample.get('sbytes', 0) <= 5.0014801672659814e-05:
                                            return 8
                                        else:
                                            return 5
                                    else:
                                        if sample.get('sbytes', 0) <= 6.478240493379417e-06:
                                            return 9
                                        else:
                                            return 3
                                else:
                                    if sample.get('djit', 0) <= 0.00013945289538241923:
                                        if sample.get('sinpkt', 0) <= 0.00016203078848775476:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_ftp_cmd', 0) <= 0.25:
                                            return 2
                                        else:
                                            return 6
                    else:
                        if sample.get('rate', 0) <= 0.10555555298924446:
                            if sample.get('sttl', 0) <= 0.5000000002328306:
                                if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                                    if sample.get('dload', 0) <= 9.252625022782013e-05:
                                        if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                            return 7
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sbytes', 0) <= 3.72672966477694e-05:
                                            return 6
                                        else:
                                            return 3
                                else:
                                    if sample.get('dur', 0) <= 0.015114961192011833:
                                        if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dinpkt', 0) <= 0.0078119642566889524:
                                            return 6
                                        else:
                                            return 3
                            else:
                                if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                    if sample.get('ct_srv_src', 0) <= 0.15322580188512802:
                                        if sample.get('sjit', 0) <= 0.0003410062054172158:
                                            return 4
                                        else:
                                            return 7
                                    else:
                                        if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                            return 7
                                        else:
                                            return 3
                                else:
                                    if sample.get('ct_dst_src_ltm', 0) <= 0.12903225421905518:
                                        if sample.get('sbytes', 0) <= 3.385403033462353e-05:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('djit', 0) <= 0.0004697563126683235:
                                            return 7
                                        else:
                                            return 6
                        else:
                            if sample.get('sload', 0) <= 0.07687926664948463:
                                if sample.get('sinpkt', 0) <= 0.0017849187133833766:
                                    if sample.get('djit', 0) <= 0.0005017797520849854:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.016129031777381897:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sbytes', 0) <= 1.2538529972516699e-05:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('sbytes', 0) <= 4.179510028734512e-06:
                                        if sample.get('ct_srv_dst', 0) <= 0.0901639349758625:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('rate', 0) <= 0.2083333283662796:
                                            return 4
                                        else:
                                            return 0
                            else:
                                if sample.get('sbytes', 0) <= 1.3026139185967622e-05:
                                    if sample.get('dbytes', 0) <= 0.00010718039629864506:
                                        if sample.get('ct_srv_src', 0) <= 0.27419354021549225:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sload', 0) <= 0.1457858681678772:
                                            return 6
                                        else:
                                            return 3
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.2049180269241333:
                                        if sample.get('sload', 0) <= 1.888018346107856e-06:
                                            return 6
                                        else:
                                            return 5
                                    else:
                                        if sample.get('smean', 0) <= 0.06824324280023575:
                                            return 3
                                        else:
                                            return 6
                else:
                    if sample.get('sjit', 0) <= 0.0009709475270938128:
                        if sample.get('ct_flw_http_mthd', 0) <= 0.03125:
                            if sample.get('dbytes', 0) <= 2.1217761968728155e-05:
                                if sample.get('sjit', 0) <= 0.0014804545207880437:
                                    if sample.get('sbytes', 0) <= 5.886143117095344e-05:
                                        if sample.get('dload', 0) <= 0.0004506259865593165:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        return 3
                                else:
                                    if sample.get('synack', 0) <= 0.007235833210870624:
                                        if sample.get('spkts', 0) <= 0.0007515265606343746:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sinpkt', 0) <= 0.0006166440434753895:
                                            return 2
                                        else:
                                            return 3
                            else:
                                if sample.get('sttl', 0) <= 0.6196078434586525:
                                    if sample.get('sjit', 0) <= 0.006319161504507065:
                                        if sample.get('dbytes', 0) <= 6.822431168984622e-05:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dload', 0) <= 0.0014423485263250768:
                                            return 3
                                        else:
                                            return 5
                                else:
                                    if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                        if sample.get('synack', 0) <= 0.005167677532881498:
                                            return 3
                                        else:
                                            return 5
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.18852458894252777:
                                            return 5
                                        else:
                                            return 6
                        else:
                            if sample.get('spkts', 0) <= 0.005072804167866707:
                                if sample.get('sload', 0) <= 7.918595201772405e-06:
                                    if sample.get('dinpkt', 0) <= 0.0009831026545725763:
                                        if sample.get('ct_src_ltm', 0) <= 0.3898305147886276:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        return 3
                                else:
                                    if sample.get('sinpkt', 0) <= 0.0004286246548872441:
                                        if sample.get('dtcpb', 0) <= 0.6471658945083618:
                                            return 5
                                        else:
                                            return 3
                                    else:
                                        if sample.get('smean', 0) <= 0.014189188834279776:
                                            return 3
                                        else:
                                            return 5
                            else:
                                if sample.get('smean', 0) <= 0.7841216027736664:
                                    if sample.get('sinpkt', 0) <= 0.00015984343917807564:
                                        if sample.get('sload', 0) <= 0.004402671940624714:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sjit', 0) <= 0.003104174742475152:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.13934426009655:
                                        if sample.get('dload', 0) <= 8.30030876386445e-05:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dloss', 0) <= 0.02151806838810444:
                                            return 9
                                        else:
                                            return 3
                    else:
                        if sample.get('sttl', 0) <= 0.6196078434586525:
                            if sample.get('sload', 0) <= 0.015761255752295256:
                                if sample.get('proto', 0) <= 0.257692314684391:
                                    if sample.get('proto', 0) <= 0.08076923154294491:
                                        if sample.get('proto', 0) <= 0.04615384619683027:
                                            return 1
                                        else:
                                            return 3
                                    else:
                                        if sample.get('stcpb', 0) <= 0.3683161288499832:
                                            return 3
                                        else:
                                            return 5
                                else:
                                    if sample.get('proto', 0) <= 0.7807692289352417:
                                        if sample.get('djit', 0) <= 0.0002738285984378308:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sloss', 0) <= 0.00028200788801768795:
                                            return 6
                                        else:
                                            return 4
                            else:
                                if sample.get('dpkts', 0) <= 0.0016336903208866715:
                                    if sample.get('ct_srv_src', 0) <= 0.4193548411130905:
                                        if sample.get('dtcpb', 0) <= 0.09416872262954712:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dmean', 0) <= 0.11733333766460419:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('dtcpb', 0) <= 0.008261153940111399:
                                        if sample.get('sinpkt', 0) <= 0.00023034722835291177:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dload', 0) <= 0.0002281865236000158:
                                            return 6
                                        else:
                                            return 2
                        else:
                            if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                                    if sample.get('service', 0) <= 0.2916666716337204:
                                        if sample.get('dbytes', 0) <= 2.1217761968728155e-05:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('rate', 0) <= 1.3703732747671893e-05:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('rate', 0) <= 1.5768399862281512e-05:
                                        if sample.get('dinpkt', 0) <= 0.0013293795636855066:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('sbytes', 0) <= 0.013257022481411695:
                                            return 3
                                        else:
                                            return 6
                            else:
                                if sample.get('sinpkt', 0) <= 0.0074192380998283625:
                                    if sample.get('sload', 0) <= 0.024571972899138927:
                                        if sample.get('dmean', 0) <= 0.8876666724681854:
                                            return 3
                                        else:
                                            return 5
                                    else:
                                        if sample.get('dloss', 0) <= 0.05874341540038586:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('rate', 0) <= 0.09545454755425453:
                                        if sample.get('ct_srv_dst', 0) <= 0.5819672048091888:
                                            return 4
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sjit', 0) <= 0.36565767228603363:
                                            return 5
                                        else:
                                            return 6
            else:
                if sample.get('dinpkt', 0) <= 0.02714757714420557:
                    if sample.get('sbytes', 0) <= 0.004707277752459049:
                        if sample.get('sjit', 0) <= 0.0006550030957441777:
                            if sample.get('sbytes', 0) <= 6.826532853665412e-06:
                                if sample.get('rate', 0) <= 2.7445148589322343e-05:
                                    if sample.get('tcprtt', 0) <= 0.01985285710543394:
                                        if sample.get('sloss', 0) <= 0.00028200788801768795:
                                            return 3
                                        else:
                                            return 7
                                    else:
                                        return 7
                                else:
                                    if sample.get('synack', 0) <= 0.008938145358115435:
                                        if sample.get('synack', 0) <= 0.00281161954626441:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 7
                                        else:
                                            return 6
                            else:
                                if sample.get('service', 0) <= 0.1250000037252903:
                                    if sample.get('spkts', 0) <= 0.0015030530630610883:
                                        if sample.get('ct_dst_ltm', 0) <= 0.06034482829272747:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        return 3
                                else:
                                    if sample.get('ct_state_ttl', 0) <= 0.1666666716337204:
                                        return 6
                                    else:
                                        return 4
                        else:
                            if sample.get('sbytes', 0) <= 8.839663496473804e-05:
                                if sample.get('spkts', 0) <= 0.0015030530630610883:
                                    if sample.get('response_body_len', 0) <= 0.0045413970947265625:
                                        if sample.get('sload', 0) <= 0.024171093478798866:
                                            return 8
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dloss', 0) <= 0.004448883235454559:
                                            return 2
                                        else:
                                            return 3
                                else:
                                    if sample.get('sbytes', 0) <= 6.011528421367984e-05:
                                        if sample.get('sjit', 0) <= 0.0017050392925739288:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('sttl', 0) <= 0.6196078434586525:
                                            return 3
                                        else:
                                            return 4
                            else:
                                if sample.get('sloss', 0) <= 0.0008460236713290215:
                                    if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                                        if sample.get('spkts', 0) <= 0.0020666979253292084:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('sinpkt', 0) <= 0.001000419957563281:
                                            return 6
                                        else:
                                            return 7
                                else:
                                    if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                        if sample.get('dmean', 0) <= 0.04033333249390125:
                                            return 4
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dtcpb', 0) <= 0.38480085134506226:
                                            return 3
                                        else:
                                            return 7
                    else:
                        if sample.get('service', 0) <= 0.3333333283662796:
                            if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                if sample.get('sloss', 0) <= 0.02256063185632229:
                                    if sample.get('dur', 0) <= 0.028396029956638813:
                                        if sample.get('is_ftp_login', 0) <= 0.25:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                            return 4
                                        else:
                                            return 4
                                else:
                                    if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                        if sample.get('sload', 0) <= 7.686047138122376e-05:
                                            return 3
                                        else:
                                            return 4
                                    else:
                                        if sample.get('dloss', 0) <= 0.0002723806173889898:
                                            return 6
                                        else:
                                            return 4
                            else:
                                if sample.get('sinpkt', 0) <= 0.005116604967042804:
                                    if sample.get('spkts', 0) <= 0.001127289782743901:
                                        if sample.get('stcpb', 0) <= 0.06509089283645153:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('djit', 0) <= 0.005257618380710483:
                                            return 6
                                        else:
                                            return 3
                                else:
                                    if sample.get('dpkts', 0) <= 0.0013614085619337857:
                                        if sample.get('spkts', 0) <= 0.011648661457002163:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        return 3
                        else:
                            if sample.get('dloss', 0) <= 0.0011803159140981734:
                                if sample.get('sjit', 0) <= 0.000954592484049499:
                                    if sample.get('rate', 0) <= 5.523774234461598e-05:
                                        if sample.get('dtcpb', 0) <= 0.1867629736661911:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        return 5
                                else:
                                    if sample.get('sload', 0) <= 3.210434922351624e-07:
                                        if sample.get('rate', 0) <= 7.506341353291646e-05:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dload', 0) <= 0.00135053473059088:
                                            return 4
                                        else:
                                            return 3
                            else:
                                if sample.get('sloss', 0) <= 0.017296484671533108:
                                    if sample.get('smean', 0) <= 0.7638513445854187:
                                        if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        return 3
                                else:
                                    if sample.get('djit', 0) <= 8.85271692823153e-05:
                                        return 4
                                    else:
                                        if sample.get('sbytes', 0) <= 0.0018016474205069244:
                                            return 6
                                        else:
                                            return 4
                else:
                    if sample.get('service', 0) <= 0.0833333358168602:
                        if sample.get('smean', 0) <= 0.04493243247270584:
                            if sample.get('sinpkt', 0) <= 0.001148318115156144:
                                if sample.get('dtcpb', 0) <= 0.16909731179475784:
                                    if sample.get('ct_src_dport_ltm', 0) <= 0.043103449046611786:
                                        if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('smean', 0) <= 0.03885135240852833:
                                            return 5
                                        else:
                                            return 3
                                else:
                                    if sample.get('ackdat', 0) <= 0.0037409800570458174:
                                        if sample.get('tcprtt', 0) <= 0.04471308924257755:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        return 5
                            else:
                                if sample.get('dbytes', 0) <= 0.00010397385631222278:
                                    if sample.get('rate', 0) <= 0.00019474924192763865:
                                        if sample.get('ct_srv_src', 0) <= 0.04838709533214569:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('rate', 0) <= 1.8994919628312346e-05:
                                            return 3
                                        else:
                                            return 3
                                else:
                                    if sample.get('synack', 0) <= 0.07570206001400948:
                                        return 3
                                    else:
                                        if sample.get('dur', 0) <= 0.016550502739846706:
                                            return 2
                                        else:
                                            return 3
                        else:
                            if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                if sample.get('sjit', 0) <= 0.006572664016857743:
                                    if sample.get('stcpb', 0) <= 0.8498727977275848:
                                        if sample.get('smean', 0) <= 0.09189189225435257:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dtcpb', 0) <= 0.8446642160415649:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('tcprtt', 0) <= 0.07720442116260529:
                                        if sample.get('ct_dst_ltm', 0) <= 0.05172413866966963:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                            return 5
                                        else:
                                            return 3
                            else:
                                if sample.get('sloss', 0) <= 0.0004700131685240194:
                                    if sample.get('sinpkt', 0) <= 0.002062960178591311:
                                        if sample.get('sload', 0) <= 1.1283521814675623e-06:
                                            return 5
                                        else:
                                            return 4
                                    else:
                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                            return 3
                                        else:
                                            return 4
                                else:
                                    if sample.get('smean', 0) <= 0.8439189195632935:
                                        if sample.get('dinpkt', 0) <= 0.0006985986838117242:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('rate', 0) <= 1.2299661193537759e-05:
                                            return 3
                                        else:
                                            return 3
                    else:
                        if sample.get('dbytes', 0) <= 0.00010042619032901712:
                            if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                if sample.get('service', 0) <= 0.2916666716337204:
                                    if sample.get('sinpkt', 0) <= 0.008604558184742928:
                                        if sample.get('sloss', 0) <= 0.001598044706042856:
                                            return 6
                                        else:
                                            return 5
                                    else:
                                        if sample.get('tcprtt', 0) <= 0.03351319395005703:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('sjit', 0) <= 0.0050534752663224936:
                                        return 3
                                    else:
                                        if sample.get('response_body_len', 0) <= 1.8215179807157256e-05:
                                            return 3
                                        else:
                                            return 6
                            else:
                                if sample.get('smean', 0) <= 0.03175675682723522:
                                    if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                        if sample.get('sjit', 0) <= 0.00158002256648615:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dbytes', 0) <= 9.906170453177765e-05:
                                            return 6
                                        else:
                                            return 3
                                else:
                                    if sample.get('sttl', 0) <= 0.6196078434586525:
                                        if sample.get('dtcpb', 0) <= 0.1645539030432701:
                                            return 6
                                        else:
                                            return 7
                                    else:
                                        if sample.get('dloss', 0) <= 0.0009079353476408869:
                                            return 4
                                        else:
                                            return 6
                        else:
                            if sample.get('dinpkt', 0) <= 0.007575789233669639:
                                if sample.get('smean', 0) <= 0.040878377854824066:
                                    if sample.get('dinpkt', 0) <= 0.0001403617497999221:
                                        if sample.get('smean', 0) <= 0.030743243172764778:
                                            return 0
                                        else:
                                            return 2
                                    else:
                                        if sample.get('ct_flw_http_mthd', 0) <= 0.875:
                                            return 3
                                        else:
                                            return 5
                                else:
                                    if sample.get('tcprtt', 0) <= 0.01373661682009697:
                                        if sample.get('dload', 0) <= 0.041898710653185844:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dur', 0) <= 0.004327450878918171:
                                            return 3
                                        else:
                                            return 7
                            else:
                                if sample.get('dpkts', 0) <= 0.012524959165602922:
                                    if sample.get('ct_srv_src', 0) <= 0.08870967477560043:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                            return 4
                                        else:
                                            return 4
                                else:
                                    if sample.get('djit', 0) <= 0.002408887492492795:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                            return 2
                                        else:
                                            return 7
                                    else:
                                        if sample.get('dload', 0) <= 0.05160294659435749:
                                            return 3
                                        else:
                                            return 3
    else:
        if sample.get('smean', 0) <= 0.022635134868323803:
            if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                if sample.get('proto', 0) <= 0.7923076748847961:
                    if sample.get('rate', 0) <= 2.4317291007491804e-06:
                        if sample.get('sload', 0) <= 0.011568992398679256:
                            if sample.get('sbytes', 0) <= 9.543214218865614e-06:
                                if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                    if sample.get('dpkts', 0) <= 0.002178253751480952:
                                        if sample.get('ct_srv_dst', 0) <= 0.07377049326896667:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sbytes', 0) <= 5.851313972016214e-06:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('ct_src_ltm', 0) <= 0.09322033822536469:
                                        if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        return 4
                            else:
                                if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                    if sample.get('dur', 0) <= 7.891668303727783e-06:
                                        if sample.get('rate', 0) <= 0.036375660449266434:
                                            return 2
                                        else:
                                            return 4
                                    else:
                                        if sample.get('sinpkt', 0) <= 1.583069639821133e-07:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('rate', 0) <= 0.1964285671710968:
                                        if sample.get('dmean', 0) <= 0.03733333200216293:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                            return 3
                                        else:
                                            return 6
                        else:
                            if sample.get('rate', 0) <= 0.10555555298924446:
                                if sample.get('sjit', 0) <= 0.0007249924237839878:
                                    if sample.get('rate', 0) <= 0.11688311398029327:
                                        if sample.get('sbytes', 0) <= 3.817285869445186e-05:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        if sample.get('rate', 0) <= 0.1805555522441864:
                                            return 4
                                        else:
                                            return 6
                                else:
                                    if sample.get('dmean', 0) <= 0.029666665941476822:
                                        if sample.get('smean', 0) <= 0.036486485973000526:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        return 2
                            else:
                                if sample.get('ct_srv_src', 0) <= 0.10483870655298233:
                                    if sample.get('sbytes', 0) <= 5.78165554543375e-06:
                                        if sample.get('dinpkt', 0) <= 0.003960955189540982:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('dinpkt', 0) <= 0.0015328407171182334:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('sload', 0) <= 8.21199209894985e-05:
                                        if sample.get('spkts', 0) <= 0.0024424612056463957:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dur', 0) <= 0.03408995643258095:
                                            return 3
                                        else:
                                            return 4
                    else:
                        if sample.get('sload', 0) <= 8.965511142378091e-07:
                            if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                if sample.get('sbytes', 0) <= 4.980582616553875e-06:
                                    if sample.get('sttl', 0) <= 0.5000000002328306:
                                        if sample.get('sttl', 0) <= 0.6196078434586525:
                                            return 3
                                        else:
                                            return 7
                                    else:
                                        return 3
                                else:
                                    if sample.get('sload', 0) <= 0.0063275115098804235:
                                        if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                            return 7
                                        else:
                                            return 5
                                    else:
                                        if sample.get('ct_src_dport_ltm', 0) <= 0.11206896603107452:
                                            return 4
                                        else:
                                            return 6
                            else:
                                if sample.get('rate', 0) <= 0.09545454755425453:
                                    if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                        if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_dst_ltm', 0) <= 0.04310344811528921:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                        return 6
                                    else:
                                        return 2
                        else:
                            if sample.get('rate', 0) <= 0.1180555522441864:
                                if sample.get('smean', 0) <= 0.016891891602426767:
                                    if sample.get('rate', 0) <= 0.02517061002436094:
                                        if sample.get('sinpkt', 0) <= 3.4994172892766073e-07:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.06557377055287361:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('sbytes', 0) <= 4.2143391283389064e-06:
                                        if sample.get('ct_src_ltm', 0) <= 0.16949152201414108:
                                            return 6
                                        else:
                                            return 2
                                    else:
                                        if sample.get('smean', 0) <= 0.016891891602426767:
                                            return 3
                                        else:
                                            return 4
                            else:
                                if sample.get('ct_dst_src_ltm', 0) <= 0.08870967477560043:
                                    if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_src_ltm', 0) <= 0.0762711875140667:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('dmean', 0) <= 0.029666665941476822:
                                        if sample.get('spkts', 0) <= 0.0376702694920823:
                                            return 6
                                        else:
                                            return 2
                                    else:
                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.05405405256897211:
                                            return 6
                                        else:
                                            return 3
                else:
                    if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                        if sample.get('sload', 0) <= 2.1128186489249856e-07:
                            if sample.get('ct_srv_dst', 0) <= 0.17213114351034164:
                                if sample.get('smean', 0) <= 0.016554053872823715:
                                    if sample.get('ct_state_ttl', 0) <= 0.4166666716337204:
                                        return 6
                                    else:
                                        if sample.get('sload', 0) <= 0.005653344094753265:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('dmean', 0) <= 0.029666665941476822:
                                        if sample.get('smean', 0) <= 0.042229728773236275:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        return 4
                            else:
                                if sample.get('stcpb', 0) <= 0.7717174589633942:
                                    if sample.get('dload', 0) <= 0.00012609863915713504:
                                        if sample.get('ackdat', 0) <= 0.015889049973338842:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('rate', 0) <= 0.04701056343037635:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('ackdat', 0) <= 0.021577770821750164:
                                        if sample.get('stcpb', 0) <= 0.8966863751411438:
                                            return 6
                                        else:
                                            return 6
                                    else:
                                        return 4
                        else:
                            if sample.get('ct_dst_src_ltm', 0) <= 0.24193548411130905:
                                if sample.get('ct_state_ttl', 0) <= 0.5833333432674408:
                                    if sample.get('dur', 0) <= 4.5000007276030374e-07:
                                        if sample.get('ct_srv_dst', 0) <= 0.0901639349758625:
                                            return 4
                                        else:
                                            return 3
                                    else:
                                        return 3
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.016393441706895828:
                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.06756756640970707:
                                            return 2
                                        else:
                                            return 3
                            else:
                                if sample.get('ct_src_ltm', 0) <= 0.1355932205915451:
                                    if sample.get('rate', 0) <= 7.591218309244141e-05:
                                        if sample.get('ct_src_dport_ltm', 0) <= 0.043103449046611786:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sload', 0) <= 0.03163755778223276:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('dur', 0) <= 1.1666668875420783e-07:
                                        if sample.get('ct_src_ltm', 0) <= 0.1355932168662548:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        return 4
                    else:
                        if sample.get('smean', 0) <= 0.04358108155429363:
                            if sample.get('sbytes', 0) <= 5.126865653437562e-05:
                                if sample.get('dmean', 0) <= 0.0566666666418314:
                                    if sample.get('sinpkt', 0) <= 0.0008648846705909818:
                                        if sample.get('dload', 0) <= 0.0003203186424798332:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('synack', 0) <= 0.05222143419086933:
                                            return 7
                                        else:
                                            return 3
                                else:
                                    if sample.get('smean', 0) <= 0.026689188554883003:
                                        if sample.get('ct_src_ltm', 0) <= 0.11016948893666267:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('synack', 0) <= 0.025035577826201916:
                                            return 4
                                        else:
                                            return 6
                            else:
                                if sample.get('sbytes', 0) <= 5.802553096145857e-05:
                                    if sample.get('dtcpb', 0) <= 0.4982120990753174:
                                        if sample.get('sload', 0) <= 3.3516239028585915e-07:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                            return 6
                                        else:
                                            return 3
                                else:
                                    if sample.get('synack', 0) <= 0.026074226014316082:
                                        if sample.get('ct_dst_ltm', 0) <= 0.27586206793785095:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dload', 0) <= 0.00011460499808890745:
                                            return 7
                                        else:
                                            return 6
                        else:
                            if sample.get('dmean', 0) <= 0.029666665941476822:
                                if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                    if sample.get('sloss', 0) <= 0.0004700131685240194:
                                        if sample.get('stcpb', 0) <= 0.08528047055006027:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sjit', 0) <= 0.0008510290936101228:
                                            return 3
                                        else:
                                            return 4
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                        return 3
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 4
                                        else:
                                            return 6
                            else:
                                if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                    if sample.get('smean', 0) <= 0.05236486531794071:
                                        if sample.get('dtcpb', 0) <= 0.9098427295684814:
                                            return 3
                                        else:
                                            return 7
                                    else:
                                        return 3
                                else:
                                    if sample.get('smean', 0) <= 0.08141892030835152:
                                        if sample.get('synack', 0) <= 0.019506240263581276:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('djit', 0) <= 0.001429394178558141:
                                            return 4
                                        else:
                                            return 3
            else:
                if sample.get('smean', 0) <= 0.02128378301858902:
                    if sample.get('ct_src_dport_ltm', 0) <= 0.16379310190677643:
                        if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                            if sample.get('ct_src_dport_ltm', 0) <= 0.06034482829272747:
                                if sample.get('ct_srv_src', 0) <= 0.18548386543989182:
                                    if sample.get('rate', 0) <= 1.5312265077227494e-05:
                                        if sample.get('sttl', 0) <= 0.1823529414832592:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        if sample.get('rate', 0) <= 0.08571428433060646:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('proto', 0) <= 0.8153845965862274:
                                        if sample.get('rate', 0) <= 0.4166666567325592:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        if sample.get('rate', 0) <= 0.3500000014901161:
                                            return 2
                                        else:
                                            return 3
                            else:
                                if sample.get('sttl', 0) <= 0.6196078434586525:
                                    return 3
                                else:
                                    if sample.get('sbytes', 0) <= 6.0184944231878035e-05:
                                        return 5
                                    else:
                                        return 4
                        else:
                            if sample.get('dload', 0) <= 0.00045721310016233474:
                                if sample.get('proto', 0) <= 0.8846153616905212:
                                    if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                        if sample.get('rate', 0) <= 8.417255685344571e-06:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        return 3
                                else:
                                    if sample.get('ct_srv_src', 0) <= 0.08870967477560043:
                                        if sample.get('sbytes', 0) <= 1.0100482086272677e-05:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                            return 2
                                        else:
                                            return 3
                            else:
                                if sample.get('spkts', 0) <= 0.00014091123011894524:
                                    if sample.get('smean', 0) <= 0.8327702581882477:
                                        return 6
                                    else:
                                        return 4
                                else:
                                    if sample.get('sload', 0) <= 0.19893696531653404:
                                        if sample.get('sbytes', 0) <= 1.9225745745643508e-05:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        return 3
                    else:
                        if sample.get('ct_srv_src', 0) <= 0.13709677010774612:
                            if sample.get('proto', 0) <= 0.9038461446762085:
                                if sample.get('smean', 0) <= 0.062162162736058235:
                                    if sample.get('ct_srv_dst', 0) <= 0.1147540956735611:
                                        if sample.get('sttl', 0) <= 0.6196078434586525:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sload', 0) <= 0.005653344094753265:
                                            return 4
                                        else:
                                            return 6
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                        if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.18548386543989182:
                                            return 6
                                        else:
                                            return 4
                            else:
                                if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                                    if sample.get('sload', 0) <= 0.01602969691157341:
                                        if sample.get('ct_srv_dst', 0) <= 0.0901639349758625:
                                            return 7
                                        else:
                                            return 5
                                    else:
                                        if sample.get('rate', 0) <= 0.05256410501897335:
                                            return 6
                                        else:
                                            return 5
                                else:
                                    if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                                        if sample.get('rate', 0) <= 0.10795454680919647:
                                            return 2
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sbytes', 0) <= 1.2817163678846555e-05:
                                            return 3
                                        else:
                                            return 6
                        else:
                            if sample.get('ct_src_ltm', 0) <= 0.05084745865315199:
                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                    return 3
                                else:
                                    if sample.get('ct_srv_src', 0) <= 0.016129031777381897:
                                        return 2
                                    else:
                                        return 3
                            else:
                                if sample.get('ct_src_ltm', 0) <= 0.0762711875140667:
                                    if sample.get('smean', 0) <= 0.19729729555547237:
                                        if sample.get('ct_srv_dst', 0) <= 0.13934426009655:
                                            return 3
                                        else:
                                            return 4
                                    else:
                                        return 3
                                else:
                                    if sample.get('rate', 0) <= 0.4166666567325592:
                                        if sample.get('proto', 0) <= 0.05769230891019106:
                                            return 4
                                        else:
                                            return 3
                                    else:
                                        if sample.get('rate', 0) <= 0.75:
                                            return 3
                                        else:
                                            return 2
                else:
                    if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                        if sample.get('sload', 0) <= 8.965511142378091e-07:
                            if sample.get('smean', 0) <= 0.0260135130956769:
                                if sample.get('dmean', 0) <= 0.03299999888986349:
                                    if sample.get('ct_dst_ltm', 0) <= 0.043103449046611786:
                                        if sample.get('sload', 0) <= 3.3268813695030985e-06:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('service', 0) <= 0.375:
                                            return 4
                                        else:
                                            return 6
                                else:
                                    if sample.get('sload', 0) <= 8.840470854920568e-06:
                                        if sample.get('tcprtt', 0) <= 0.0294546727091074:
                                            return 3
                                        else:
                                            return 0
                                    else:
                                        if sample.get('sinpkt', 0) <= 0.0005807208945043385:
                                            return 3
                                        else:
                                            return 4
                            else:
                                if sample.get('sbytes', 0) <= 7.592776455567218e-06:
                                    if sample.get('tcprtt', 0) <= 0.04849344491958618:
                                        if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('dbytes', 0) <= 4.584674024954438e-05:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('rate', 0) <= 0.07878788188099861:
                                        if sample.get('dbytes', 0) <= 4.7347673898912035e-05:
                                            return 3
                                        else:
                                            return 4
                                    else:
                                        if sample.get('smean', 0) <= 0.05574324354529381:
                                            return 2
                                        else:
                                            return 3
                        else:
                            if sample.get('dmean', 0) <= 0.19033332914113998:
                                if sample.get('dur', 0) <= 0.004199517425149679:
                                    if sample.get('sjit', 0) <= 0.0007882077170506818:
                                        if sample.get('dmean', 0) <= 0.03266666550189257:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        return 3
                                else:
                                    if sample.get('dur', 0) <= 0.00508850091136992:
                                        return 6
                                    else:
                                        if sample.get('tcprtt', 0) <= 0.09540176391601562:
                                            return 4
                                        else:
                                            return 2
                            else:
                                if sample.get('service', 0) <= 0.0833333358168602:
                                    if sample.get('ct_src_ltm', 0) <= 0.08474576473236084:
                                        if sample.get('dur', 0) <= 0.003153683850541711:
                                            return 2
                                        else:
                                            return 6
                                    else:
                                        if sample.get('tcprtt', 0) <= 0.009126604534685612:
                                            return 3
                                        else:
                                            return 5
                                else:
                                    if sample.get('dloss', 0) <= 0.0013619030360132456:
                                        if sample.get('tcprtt', 0) <= 0.0033865284640341997:
                                            return 5
                                        else:
                                            return 3
                                    else:
                                        if sample.get('smean', 0) <= 0.04560810700058937:
                                            return 3
                                        else:
                                            return 5
                    else:
                        if sample.get('dloss', 0) <= 0.0011803159140981734:
                            if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                if sample.get('synack', 0) <= 0.03768840618431568:
                                    if sample.get('rate', 0) <= 2.064591899397783e-05:
                                        if sample.get('dinpkt', 0) <= 0.0017434571054764092:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        return 3
                                else:
                                    if sample.get('dur', 0) <= 0.04586028307676315:
                                        if sample.get('dload', 0) <= 0.00031717978708911687:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        return 3
                            else:
                                if sample.get('dmean', 0) <= 0.07066666707396507:
                                    if sample.get('spkts', 0) <= 0.001784875465091318:
                                        if sample.get('sjit', 0) <= 0.0047205230221152306:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        return 4
                                else:
                                    if sample.get('djit', 0) <= 0.0003470317169558257:
                                        return 3
                                    else:
                                        if sample.get('dpkts', 0) <= 0.001542929734569043:
                                            return 6
                                        else:
                                            return 2
                        else:
                            if sample.get('sttl', 0) <= 0.6215686276555061:
                                if sample.get('dload', 0) <= 0.03968756087124348:
                                    if sample.get('ct_flw_http_mthd', 0) <= 0.15625:
                                        return 3
                                    else:
                                        return 7
                                else:
                                    if sample.get('dmean', 0) <= 0.8993333280086517:
                                        if sample.get('sbytes', 0) <= 0.002041760249994695:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        if sample.get('smean', 0) <= 0.014864864759147167:
                                            return 3
                                        else:
                                            return 5
                            else:
                                if sample.get('dbytes', 0) <= 0.00015179910406004637:
                                    if sample.get('rate', 0) <= 6.406802413039259e-06:
                                        return 3
                                    else:
                                        if sample.get('smean', 0) <= 0.0996621623635292:
                                            return 2
                                        else:
                                            return 5
                                else:
                                    if sample.get('sload', 0) <= 1.7004263668241038e-06:
                                        if sample.get('rate', 0) <= 7.21733522368595e-05:
                                            return 5
                                        else:
                                            return 9
                                    else:
                                        if sample.get('dloss', 0) <= 0.00798983103595674:
                                            return 3
                                        else:
                                            return 4
        else:
            if sample.get('ct_state_ttl', 0) <= 0.6666666716337204:
                if sample.get('ct_dst_sport_ltm', 0) <= 0.14864864200353622:
                    if sample.get('dpkts', 0) <= 0.000998366333078593:
                        if sample.get('ct_flw_http_mthd', 0) <= 0.15625:
                            if sample.get('synack', 0) <= 0.011209908872842789:
                                if sample.get('smean', 0) <= 0.04763513617217541:
                                    if sample.get('ct_src_ltm', 0) <= 0.08474576100707054:
                                        if sample.get('smean', 0) <= 0.016554054338485003:
                                            return 3
                                        else:
                                            return 7
                                    else:
                                        if sample.get('sload', 0) <= 0.02982969768345356:
                                            return 0
                                        else:
                                            return 6
                                else:
                                    if sample.get('sload', 0) <= 8.461914148938376e-06:
                                        if sample.get('sloss', 0) <= 0.0004700131685240194:
                                            return 5
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dpkts', 0) <= 0.0002722817152971402:
                                            return 2
                                        else:
                                            return 6
                            else:
                                if sample.get('smean', 0) <= 0.05304053984582424:
                                    if sample.get('dload', 0) <= 0.0005112524668220431:
                                        if sample.get('sjit', 0) <= 0.005211434559896588:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.16935483366250992:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('rate', 0) <= 0.2499999850988388:
                                        if sample.get('ct_dst_ltm', 0) <= 0.1982758641242981:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_dst_ltm', 0) <= 0.12068965658545494:
                                            return 6
                                        else:
                                            return 3
                        else:
                            if sample.get('rate', 0) <= 0.1458333283662796:
                                if sample.get('smean', 0) <= 0.021283783949911594:
                                    if sample.get('rate', 0) <= 2.21659956878284e-05:
                                        if sample.get('smean', 0) <= 0.023310811258852482:
                                            return 6
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dbytes', 0) <= 3.984300019510556e-05:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('dload', 0) <= 0.0001899798953672871:
                                        if sample.get('dur', 0) <= 0.012522560544312:
                                            return 8
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sjit', 0) <= 0.0015179040492512286:
                                            return 1
                                        else:
                                            return 6
                            else:
                                if sample.get('proto', 0) <= 0.257692314684391:
                                    if sample.get('dur', 0) <= 0.020421787165105343:
                                        if sample.get('proto', 0) <= 0.08076923154294491:
                                            return 7
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dload', 0) <= 4.516112858254928e-05:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('dload', 0) <= 9.715293344925158e-05:
                                        if sample.get('response_body_len', 0) <= 3.080368151131552e-05:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('stcpb', 0) <= 0.16881956160068512:
                                            return 3
                                        else:
                                            return 0
                    else:
                        if sample.get('dbytes', 0) <= 0.00010124488107976504:
                            if sample.get('sbytes', 0) <= 1.2538529972516699e-05:
                                if sample.get('smean', 0) <= 0.049662161618471146:
                                    if sample.get('djit', 0) <= 5.3953550377627835e-05:
                                        if sample.get('smean', 0) <= 0.040878377854824066:
                                            return 7
                                        else:
                                            return 2
                                    else:
                                        if sample.get('dinpkt', 0) <= 0.0012648910051211715:
                                            return 3
                                        else:
                                            return 7
                                else:
                                    if sample.get('tcprtt', 0) <= 0.06214802339673042:
                                        if sample.get('smean', 0) <= 0.042229728773236275:
                                            return 3
                                        else:
                                            return 4
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.12295081466436386:
                                            return 6
                                        else:
                                            return 7
                            else:
                                if sample.get('service', 0) <= 0.0416666679084301:
                                    if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                        if sample.get('sbytes', 0) <= 3.246086089347955e-05:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sbytes', 0) <= 0.00016592653992120177:
                                            return 6
                                        else:
                                            return 3
                                else:
                                    if sample.get('ct_srv_src', 0) <= 0.06451612710952759:
                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.04054053872823715:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sinpkt', 0) <= 0.002135905670002103:
                                            return 4
                                        else:
                                            return 6
                        else:
                            if sample.get('ct_flw_http_mthd', 0) <= 0.03125:
                                if sample.get('dpkts', 0) <= 0.11753493919968605:
                                    if sample.get('dbytes', 0) <= 0.0001782701292540878:
                                        if sample.get('djit', 0) <= 0.0005076527741039172:
                                            return 4
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.10810810327529907:
                                            return 4
                                        else:
                                            return 3
                                else:
                                    if sample.get('dbytes', 0) <= 8.671310206409544e-05:
                                        if sample.get('dmean', 0) <= 0.035333333536982536:
                                            return 1
                                        else:
                                            return 2
                                    else:
                                        if sample.get('sbytes', 0) <= 6.589694021386094e-05:
                                            return 6
                                        else:
                                            return 3
                            else:
                                if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                    if sample.get('ct_srv_src', 0) <= 0.16129031777381897:
                                        if sample.get('djit', 0) <= 0.0002966881147585809:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                            return 4
                                        else:
                                            return 6
                                else:
                                    if sample.get('sload', 0) <= 6.251701506698737e-06:
                                        if sample.get('rate', 0) <= 0.1964285671710968:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('smean', 0) <= 0.06182432360947132:
                                            return 3
                                        else:
                                            return 4
                else:
                    if sample.get('ct_src_dport_ltm', 0) <= 0.16379310190677643:
                        if sample.get('ct_dst_ltm', 0) <= 0.1465517207980156:
                            if sample.get('sbytes', 0) <= 9.96116523310775e-06:
                                if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                    if sample.get('dtcpb', 0) <= 0.7036108374595642:
                                        if sample.get('dur', 0) <= 0.0004087750848924543:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sbytes', 0) <= 8.985946351458551e-06:
                                            return 7
                                        else:
                                            return 8
                                else:
                                    if sample.get('proto', 0) <= 0.8153845965862274:
                                        return 3
                                    else:
                                        if sample.get('sbytes', 0) <= 8.010727469809353e-06:
                                            return 2
                                        else:
                                            return 6
                            else:
                                if sample.get('ct_src_dport_ltm', 0) <= 0.060344829224050045:
                                    if sample.get('ct_state_ttl', 0) <= 0.4166666716337204:
                                        return 7
                                    else:
                                        return 3
                                else:
                                    if sample.get('sloss', 0) <= 0.00028200788801768795:
                                        return 3
                                    else:
                                        if sample.get('dur', 0) <= 0.010691259987652302:
                                            return 3
                                        else:
                                            return 6
                        else:
                            if sample.get('smean', 0) <= 0.033445945009589195:
                                if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                    if sample.get('smean', 0) <= 0.021283783949911594:
                                        if sample.get('smean', 0) <= 0.019256756640970707:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 3
                                        else:
                                            return 3
                                else:
                                    if sample.get('ct_state_ttl', 0) <= 0.6666666716337204:
                                        if sample.get('ct_src_ltm', 0) <= 0.05932203307747841:
                                            return 8
                                        else:
                                            return 4
                                    else:
                                        if sample.get('sbytes', 0) <= 3.7406613046186976e-05:
                                            return 6
                                        else:
                                            return 3
                            else:
                                if sample.get('proto', 0) <= 0.04615384619683027:
                                    if sample.get('sbytes', 0) <= 1.2259895811439492e-05:
                                        if sample.get('sbytes', 0) <= 1.2538529972516699e-05:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('proto', 0) <= 0.9038461446762085:
                                            return 2
                                        else:
                                            return 3
                                else:
                                    if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                            return 4
                                        else:
                                            return 3
                                    else:
                                        return 6
                    else:
                        if sample.get('dpkts', 0) <= 0.330731526017189:
                            if sample.get('sloss', 0) <= 0.0010340289736632258:
                                if sample.get('dmean', 0) <= 0.887333333492279:
                                    if sample.get('proto', 0) <= 0.12692307867109776:
                                        if sample.get('proto', 0) <= 0.04615384619683027:
                                            return 2
                                        else:
                                            return 4
                                    else:
                                        return 4
                                else:
                                    if sample.get('ct_src_ltm', 0) <= 0.11016948893666267:
                                        if sample.get('ct_srv_src', 0) <= 0.13709677010774612:
                                            return 7
                                        else:
                                            return 3
                                    else:
                                        if sample.get('smean', 0) <= 0.08006756752729416:
                                            return 3
                                        else:
                                            return 5
                            else:
                                if sample.get('proto', 0) <= 0.9038461446762085:
                                    if sample.get('ct_srv_src', 0) <= 0.20967741310596466:
                                        if sample.get('sbytes', 0) <= 8.916287697502412e-06:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dinpkt', 0) <= 0.006241530645638704:
                                            return 3
                                        else:
                                            return 1
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.1147540956735611:
                                        if sample.get('ct_srv_dst', 0) <= 0.5655737519264221:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.23770491778850555:
                                            return 6
                                        else:
                                            return 4
                        else:
                            if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                if sample.get('sbytes', 0) <= 7.362903488683514e-05:
                                    if sample.get('ct_srv_dst', 0) <= 0.06557377055287361:
                                        return 3
                                    else:
                                        return 4
                                else:
                                    if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                        if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                            return 8
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                            return 6
                                        else:
                                            return 4
                            else:
                                if sample.get('ct_srv_src', 0) <= 0.12096773833036423:
                                    if sample.get('sjit', 0) <= 0.02285133022814989:
                                        if sample.get('sjit', 0) <= 0.010450669564306736:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        return 3
                                else:
                                    if sample.get('rate', 0) <= 0.08712121099233627:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.35483870655298233:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        return 6
            else:
                if sample.get('sjit', 0) <= 0.08858245983719826:
                    if sample.get('dbytes', 0) <= 0.00010022151764133014:
                        if sample.get('dpkts', 0) <= 0.0008168451604433358:
                            if sample.get('smean', 0) <= 0.025337837636470795:
                                if sample.get('sjit', 0) <= 0.000983074656687677:
                                    if sample.get('sbytes', 0) <= 6.951918112463318e-05:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 3
                                        else:
                                            return 5
                                    else:
                                        if sample.get('dmean', 0) <= 0.03999999910593033:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('sbytes', 0) <= 4.0750221160124056e-05:
                                        if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sbytes', 0) <= 5.551782487600576e-05:
                                            return 6
                                        else:
                                            return 7
                            else:
                                if sample.get('dbytes', 0) <= 9.578693789080717e-05:
                                    if sample.get('sbytes', 0) <= 6.540933100041002e-05:
                                        if sample.get('tcprtt', 0) <= 0.06717018038034439:
                                            return 3
                                        else:
                                            return 4
                                    else:
                                        if sample.get('dmean', 0) <= 0.09600000083446503:
                                            return 7
                                        else:
                                            return 3
                                else:
                                    if sample.get('sttl', 0) <= 0.6196078434586525:
                                        if sample.get('dinpkt', 0) <= 0.06134720705449581:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dbytes', 0) <= 9.469534779782407e-05:
                                            return 7
                                        else:
                                            return 2
                        else:
                            if sample.get('sloss', 0) <= 0.0010340289736632258:
                                if sample.get('dmean', 0) <= 0.8856666684150696:
                                    if sample.get('smean', 0) <= 0.049662161618471146:
                                        if sample.get('sinpkt', 0) <= 0.002067741472274065:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dtcpb', 0) <= 0.7758811116218567:
                                            return 2
                                        else:
                                            return 3
                                else:
                                    if sample.get('dpkts', 0) <= 0.16827010363340378:
                                        if sample.get('proto', 0) <= 0.4923076927661896:
                                            return 3
                                        else:
                                            return 2
                                    else:
                                        return 3
                            else:
                                if sample.get('djit', 0) <= 9.131628394243307e-05:
                                    if sample.get('dloss', 0) <= 0.0002723806173889898:
                                        if sample.get('smean', 0) <= 0.018581081181764603:
                                            return 3
                                        else:
                                            return 1
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('dttl', 0) <= 0.9980237185955048:
                                        if sample.get('sjit', 0) <= 0.003481756546534598:
                                            return 4
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sbytes', 0) <= 5.2731484174728394e-05:
                                            return 1
                                        else:
                                            return 4
                    else:
                        if sample.get('dloss', 0) <= 0.2718358337879181:
                            if sample.get('response_body_len', 0) <= 0.004329109098762274:
                                if sample.get('ct_src_dport_ltm', 0) <= 0.043103449046611786:
                                    if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                                        if sample.get('dinpkt', 0) <= 0.00029826914285635553:
                                            return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('synack', 0) <= 0.01037982665002346:
                                            return 2
                                        else:
                                            return 3
                                else:
                                    if sample.get('ct_src_ltm', 0) <= 0.11016948893666267:
                                        return 6
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                            return 2
                                        else:
                                            return 6
                            else:
                                if sample.get('djit', 0) <= 0.00011402792733861133:
                                    if sample.get('sttl', 0) <= 0.6196078434586525:
                                        if sample.get('djit', 0) <= 0.0034981337375938892:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dmean', 0) <= 0.0976666659116745:
                                            return 2
                                        else:
                                            return 4
                                else:
                                    if sample.get('sjit', 0) <= 0.0013487301184795797:
                                        if sample.get('rate', 0) <= 0.028991595841944218:
                                            return 1
                                        else:
                                            return 4
                                    else:
                                        if sample.get('dload', 0) <= 0.0014089514152146876:
                                            return 3
                                        else:
                                            return 3
                        else:
                            if sample.get('ct_dst_ltm', 0) <= 0.10344827547669411:
                                if sample.get('sload', 0) <= 0.01602969691157341:
                                    if sample.get('response_body_len', 0) <= 0.0045413970947265625:
                                        return 3
                                    else:
                                        if sample.get('sload', 0) <= 2.3894989453765447e-06:
                                            return 3
                                        else:
                                            return 2
                                else:
                                    if sample.get('ct_dst_src_ltm', 0) <= 0.08870967477560043:
                                        return 2
                                    else:
                                        if sample.get('sjit', 0) <= 0.005240161204710603:
                                            return 3
                                        else:
                                            return 2
                            else:
                                if sample.get('ackdat', 0) <= 0.022280282340943813:
                                    if sample.get('rate', 0) <= 0.12142856791615486:
                                        if sample.get('smean', 0) <= 0.03040540497750044:
                                            return 3
                                        else:
                                            return 5
                                    else:
                                        if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                            return 1
                                        else:
                                            return 3
                                else:
                                    if sample.get('sinpkt', 0) <= 0.0012856764951720834:
                                        if sample.get('djit', 0) <= 0.0054300634656101465:
                                            return 2
                                        else:
                                            return 3
                                    else:
                                        return 1
                else:
                    if sample.get('proto', 0) <= 0.8153845965862274:
                        if sample.get('smean', 0) <= 0.042905405163764954:
                            if sample.get('dpkts', 0) <= 0.000635324016911909:
                                if sample.get('dbytes', 0) <= 0.00010424675201647915:
                                    if sample.get('response_body_len', 0) <= 2.2411346435546875e-05:
                                        if sample.get('dur', 0) <= 0.005722442641854286:
                                            return 2
                                        else:
                                            return 6
                                    else:
                                        if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                            return 6
                                        else:
                                            return 3
                                else:
                                    if sample.get('dmean', 0) <= 0.8923333287239075:
                                        if sample.get('dbytes', 0) <= 2.7016828425985295e-05:
                                            return 6
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dur', 0) <= 0.004956600954756141:
                                            return 5
                                        else:
                                            return 3
                            else:
                                if sample.get('sjit', 0) <= 0.003836262854747474:
                                    if sample.get('smean', 0) <= 0.019256756640970707:
                                        if sample.get('sjit', 0) <= 0.0015656835748814046:
                                            return 3
                                        else:
                                            return 0
                                    else:
                                        if sample.get('response_body_len', 0) <= 0.012158202938735485:
                                            return 5
                                        else:
                                            return 3
                                else:
                                    if sample.get('smean', 0) <= 0.021283783949911594:
                                        if sample.get('sbytes', 0) <= 3.740661486517638e-05:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('synack', 0) <= 0.021861832588911057:
                                            return 3
                                        else:
                                            return 3
                        else:
                            if sample.get('sjit', 0) <= 0.0032594953663647175:
                                if sample.get('sloss', 0) <= 0.004794134292751551:
                                    if sample.get('dpkts', 0) <= 0.0011798874475061893:
                                        if sample.get('djit', 0) <= 0.0024252611910924315:
                                            return 7
                                        else:
                                            return 2
                                    else:
                                        if sample.get('dbytes', 0) <= 0.00013965516700409353:
                                            return 4
                                        else:
                                            return 6
                                else:
                                    if sample.get('sbytes', 0) <= 0.004749630112200975:
                                        if sample.get('djit', 0) <= 0.00013546522677643225:
                                            return 7
                                        else:
                                            return 5
                                    else:
                                        if sample.get('ackdat', 0) <= 0.019307882525026798:
                                            return 7
                                        else:
                                            return 4
                            else:
                                if sample.get('dloss', 0) <= 0.0002723806173889898:
                                    if sample.get('service', 0) <= 0.2083333283662796:
                                        if sample.get('sbytes', 0) <= 0.000173449661815539:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('sttl', 0) <= 0.6196078434586525:
                                        if sample.get('dload', 0) <= 0.0003965431824326515:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('dloss', 0) <= 0.00063555475207977:
                                            return 6
                                        else:
                                            return 4
                    else:
                        if sample.get('sload', 0) <= 0.0734221450984478:
                            if sample.get('ct_src_dport_ltm', 0) <= 0.1465517207980156:
                                if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                                    if sample.get('spkts', 0) <= 0.0008924377907533199:
                                        if sample.get('ackdat', 0) <= 0.001112750731408596:
                                            return 4
                                        else:
                                            return 0
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                            return 3
                                        else:
                                            return 4
                                else:
                                    if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                        if sample.get('sjit', 0) <= 0.009281893260776997:
                                            return 3
                                        else:
                                            return 6
                                    else:
                                        return 2
                            else:
                                if sample.get('ct_srv_dst', 0) <= 0.46721310913562775:
                                    if sample.get('stcpb', 0) <= 0.07139570266008377:
                                        if sample.get('sjit', 0) <= 0.0046379955019801855:
                                            return 4
                                        else:
                                            return 7
                                    else:
                                        if sample.get('sloss', 0) <= 0.0004700131685240194:
                                            return 6
                                        else:
                                            return 6
                                else:
                                    if sample.get('sjit', 0) <= 0.0045639683958143:
                                        if sample.get('dmean', 0) <= 0.0663333348929882:
                                            return 4
                                        else:
                                            return 5
                                    else:
                                        if sample.get('rate', 0) <= 1.8291751985088922e-05:
                                            return 3
                                        else:
                                            return 6
                        else:
                            if sample.get('dinpkt', 0) <= 0.0011300259502604604:
                                if sample.get('ct_srv_dst', 0) <= 0.5655737519264221:
                                    if sample.get('ct_dst_sport_ltm', 0) <= 0.05405405256897211:
                                        if sample.get('tcprtt', 0) <= 0.015600430313497782:
                                            return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('sjit', 0) <= 0.0020515648648142815:
                                            return 6
                                        else:
                                            return 4
                                else:
                                    if sample.get('dbytes', 0) <= 6.774674693588167e-05:
                                        if sample.get('tcprtt', 0) <= 0.009275369811803102:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('dload', 0) <= 0.0012311747414059937:
                                            return 6
                                        else:
                                            return 4
                            else:
                                if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                    if sample.get('dbytes', 0) <= 4.9735524953575805e-05:
                                        if sample.get('dloss', 0) <= 0.0008171417866833508:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        if sample.get('tcprtt', 0) <= 0.07002576068043709:
                                            return 3
                                        else:
                                            return 7
                                else:
                                    if sample.get('dload', 0) <= 1.846669147198554e-05:
                                        if sample.get('stcpb', 0) <= 0.19812537729740143:
                                            return 6
                                        else:
                                            return 4
                                    else:
                                        if sample.get('dbytes', 0) <= 0.00013331031004781835:
                                            return 4
                                        else:
                                            return 6
