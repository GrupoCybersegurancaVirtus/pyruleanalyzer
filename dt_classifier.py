def predict(sample):
    if sample.get('ct_dst_sport_ltm', 0) <= 0.040540541522204876:
        if sample.get('sttl', 0) <= 0.239215686917305:
            return 6
        else:
            if sample.get('dbytes', 0) <= 0.00010049441698356532:
                if sample.get('ct_dst_src_ltm', 0) <= 0.008064515888690948:
                    if sample.get('sbytes', 0) <= 3.768524948100094e-05:
                        if sample.get('smean', 0) <= 0.021283783949911594:
                            if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                if sample.get('smean', 0) <= 0.018581081181764603:
                                    if sample.get('dmean', 0) <= 0.05700000002980232:
                                        if sample.get('state', 0) <= 0.75:
                                            if sample.get('dmean', 0) <= 0.03499999828636646:
                                                if sample.get('dload', 0) <= 0.00016702686116332188:
                                                    if sample.get('proto', 0) <= 0.7923076748847961:
                                                        if sample.get('proto', 0) <= 0.35384616255760193:
                                                            return 6
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('sbytes', 0) <= 3.900875867657305e-06:
                                                            if sample.get('sload', 0) <= 0.006496245041489601:
                                                                return 3
                                                            else:
                                                                if sample.get('smean', 0) <= 0.009797296952456236:
                                                                    return 3
                                                                else:
                                                                    return 6
                                                        else:
                                                            if sample.get('sbytes', 0) <= 3.350573933857959e-05:
                                                                if sample.get('dtcpb', 0) <= 0.507035031914711:
                                                                    if sample.get('smean', 0) <= 0.01722972933202982:
                                                                        if sample.get('dur', 0) <= 4.0758341128821485e-05:
                                                                            if sample.get('ct_src_ltm', 0) <= 0.16949152015149593:
                                                                                if sample.get('sbytes', 0) <= 4.492973175729276e-06:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('sbytes', 0) <= 5.0850703701144084e-06:
                                                                                        if sample.get('ct_srv_src', 0) <= 0.032258063554763794:
                                                                                            return 5
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('stcpb', 0) <= 0.021988457068800926:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('ct_srv_src', 0) <= 0.032258063554763794:
                                                                                    if sample.get('ackdat', 0) <= 0.023540875874459743:
                                                                                        return 3
                                                                                    else:
                                                                                        return 2
                                                                                else:
                                                                                    return 3
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.8554112911224365:
                                                                            if sample.get('sbytes', 0) <= 3.3227102903765626e-05:
                                                                                if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                                    return 5
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('sload', 0) <= 6.248056934055057e-07:
                                                                        if sample.get('smean', 0) <= 0.01520270248875022:
                                                                            return 7
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('synack', 0) <= 0.021833322010934353:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                            else:
                                                                if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                    if sample.get('dtcpb', 0) <= 0.9382317066192627:
                                                                        return 6
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                else:
                                                    if sample.get('smean', 0) <= 0.016554053872823715:
                                                        if sample.get('sinpkt', 0) <= 0.0011375011526979506:
                                                            if sample.get('ct_state_ttl', 0) <= 0.5833333432674408:
                                                                if sample.get('synack', 0) <= 0.007107997313141823:
                                                                    if sample.get('dmean', 0) <= 0.029333332553505898:
                                                                        return 3
                                                                    else:
                                                                        return 2
                                                                else:
                                                                    if sample.get('djit', 0) <= 0.00016080586647149175:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dload', 0) <= 0.00022178371727932245:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('synack', 0) <= 0.02167093101888895:
                                                                                return 5
                                                                            else:
                                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('djit', 0) <= 0.0002299941043020226:
                                                                return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('rate', 0) <= 5.612797940557357e-05:
                                                            if sample.get('synack', 0) <= 0.007635611807927489:
                                                                return 3
                                                            else:
                                                                if sample.get('stcpb', 0) <= 0.4643188565969467:
                                                                    if sample.get('sjit', 0) <= 0.0012797376839444041:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                        else:
                                                            if sample.get('ackdat', 0) <= 0.01623134221881628:
                                                                return 3
                                                            else:
                                                                if sample.get('tcprtt', 0) <= 0.01573166297748685:
                                                                    return 3
                                                                else:
                                                                    return 5
                                            else:
                                                if sample.get('service', 0) <= 0.2500000074505806:
                                                    if sample.get('dtcpb', 0) <= 0.5118964612483978:
                                                        if sample.get('rate', 0) <= 0.0002499319234630093:
                                                            if sample.get('stcpb', 0) <= 0.9137564301490784:
                                                                return 1
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('rate', 0) <= 3.230035235901596e-05:
                                                            return 3
                                                        else:
                                                            if sample.get('dtcpb', 0) <= 0.747245192527771:
                                                                return 3
                                                            else:
                                                                return 1
                                                else:
                                                    return 3
                                        else:
                                            return 6
                                    else:
                                        return 3
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                        if sample.get('proto', 0) <= 0.7923076748847961:
                                            return 3
                                        else:
                                            if sample.get('dload', 0) <= 0.00023436459014192224:
                                                if sample.get('service', 0) <= 0.0833333358168602:
                                                    if sample.get('dloss', 0) <= 0.0002723806173889898:
                                                        if sample.get('tcprtt', 0) <= 0.09665403142571449:
                                                            if sample.get('smean', 0) <= 0.020608107559382915:
                                                                if sample.get('sinpkt', 0) <= 0.008083099499344826:
                                                                    if sample.get('dtcpb', 0) <= 0.612118273973465:
                                                                        if sample.get('synack', 0) <= 0.03803209401667118:
                                                                            if sample.get('dload', 0) <= 0.00014814382302574813:
                                                                                if sample.get('sload', 0) <= 0.006879730615764856:
                                                                                    return 6
                                                                                else:
                                                                                    if sample.get('sload', 0) <= 0.007537791971117258:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                                            return 6
                                                                                        else:
                                                                                            if sample.get('sload', 0) <= 0.01776765286922455:
                                                                                                return 6
                                                                                            else:
                                                                                                if sample.get('rate', 0) <= 0.2916666567325592:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 6
                                                                            else:
                                                                                if sample.get('sinpkt', 0) <= 0.0013574628392234445:
                                                                                    return 6
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            if sample.get('synack', 0) <= 0.040286967530846596:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                    else:
                                                                        if sample.get('sbytes', 0) <= 3.545617619238328e-05:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                                if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                                    return 6
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('dtcpb', 0) <= 0.8128083944320679:
                                                                                    if sample.get('synack', 0) <= 0.03012407384812832:
                                                                                        if sample.get('rate', 0) <= 1.1236377076784265e-05:
                                                                                            return 3
                                                                                        else:
                                                                                            return 4
                                                                                    else:
                                                                                        return 6
                                                                                else:
                                                                                    return 6
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('ct_src_ltm', 0) <= 0.04237288236618042:
                                                        return 3
                                                    else:
                                                        return 2
                                            else:
                                                if sample.get('sbytes', 0) <= 3.559549259080086e-05:
                                                    if sample.get('sbytes', 0) <= 3.538651799317449e-05:
                                                        if sample.get('dmean', 0) <= 0.029666665941476822:
                                                            return 3
                                                        else:
                                                            if sample.get('sload', 0) <= 3.923591748389299e-06:
                                                                if sample.get('stcpb', 0) <= 0.3938722163438797:
                                                                    return 3
                                                                else:
                                                                    return 2
                                                            else:
                                                                return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('dloss', 0) <= 0.0002723806173889898:
                                                        if sample.get('dtcpb', 0) <= 0.0601680320687592:
                                                            return 3
                                                        else:
                                                            return 8
                                                    else:
                                                        return 3
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                            if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                if sample.get('dload', 0) <= 0.00011804298992501572:
                                                    if sample.get('ackdat', 0) <= 0.04396953247487545:
                                                        if sample.get('sinpkt', 0) <= 0.001715345832053572:
                                                            if sample.get('stcpb', 0) <= 0.363633468747139:
                                                                if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                    if sample.get('sload', 0) <= 0.008335441816598177:
                                                                        return 3
                                                                    else:
                                                                        return 6
                                                                else:
                                                                    if sample.get('stcpb', 0) <= 0.11298273503780365:
                                                                        if sample.get('sload', 0) <= 0.012283809017390013:
                                                                            return 4
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                if sample.get('synack', 0) <= 0.023425927385687828:
                                                                    return 3
                                                                else:
                                                                    return 4
                                                        else:
                                                            if sample.get('smean', 0) <= 0.02027027029544115:
                                                                if sample.get('synack', 0) <= 0.047103650867938995:
                                                                    return 6
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 8
                                                    else:
                                                        return 4
                                                else:
                                                    if sample.get('smean', 0) <= 0.01993243210017681:
                                                        if sample.get('dtcpb', 0) <= 0.44323693215847015:
                                                            if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                return 8
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('sbytes', 0) <= 3.46899323631078e-05:
                                                                return 3
                                                            else:
                                                                return 4
                                                    else:
                                                        if sample.get('sjit', 0) <= 0.003849177621304989:
                                                            return 3
                                                        else:
                                                            return 3
                                            else:
                                                if sample.get('ct_dst_ltm', 0) <= 0.051724137738347054:
                                                    if sample.get('dtcpb', 0) <= 0.18701231479644775:
                                                        return 3
                                                    else:
                                                        return 6
                                                else:
                                                    return 3
                                        else:
                                            if sample.get('ackdat', 0) <= 0.014701865147799253:
                                                if sample.get('smean', 0) <= 0.02027027029544115:
                                                    if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                        if sample.get('sload', 0) <= 6.017674394342976e-07:
                                                            return 3
                                                        else:
                                                            return 6
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('dtcpb', 0) <= 0.18721021711826324:
                                                    return 3
                                                else:
                                                    return 4
                            else:
                                if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                    if sample.get('sload', 0) <= 0.006879730615764856:
                                        if sample.get('dload', 0) <= 0.00021481511066667736:
                                            if sample.get('dinpkt', 0) <= 0.004062244785018265:
                                                if sample.get('smean', 0) <= 0.020608107559382915:
                                                    if sample.get('sbytes', 0) <= 3.538651799317449e-05:
                                                        if sample.get('smean', 0) <= 0.018243242986500263:
                                                            return 6
                                                        else:
                                                            return 8
                                                    else:
                                                        return 6
                                                else:
                                                    return 8
                                            else:
                                                if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                                    return 1
                                                else:
                                                    return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('rate', 0) <= 0.10555555298924446:
                                            return 4
                                        else:
                                            if sample.get('sload', 0) <= 0.010027599520981312:
                                                return 6
                                            else:
                                                return 3
                                else:
                                    if sample.get('sload', 0) <= 0.008178760763257742:
                                        if sample.get('ackdat', 0) <= 0.03193891607224941:
                                            return 4
                                        else:
                                            if sample.get('ackdat', 0) <= 0.031979719176888466:
                                                return 3
                                            else:
                                                if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                    if sample.get('djit', 0) <= 0.0005987679760437459:
                                                        return 3
                                                    else:
                                                        if sample.get('djit', 0) <= 0.0006827535689808428:
                                                            return 3
                                                        else:
                                                            if sample.get('sbytes', 0) <= 3.40630067512393e-05:
                                                                return 3
                                                            else:
                                                                return 4
                                                else:
                                                    return 4
                                    else:
                                        if sample.get('sload', 0) <= 0.00932253384962678:
                                            return 3
                                        else:
                                            if sample.get('sload', 0) <= 0.023032141849398613:
                                                return 4
                                            else:
                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                    if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                        if sample.get('ct_srv_dst', 0) <= 0.049180325120687485:
                                                            return 4
                                                        else:
                                                            return 3
                                                    else:
                                                        return 4
                                                else:
                                                    if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                        if sample.get('ct_srv_src', 0) <= 0.04838709533214569:
                                                            return 3
                                                        else:
                                                            return 6
                                                    else:
                                                        return 4
                        else:
                            if sample.get('sbytes', 0) <= 6.338923412840813e-06:
                                if sample.get('service', 0) <= 0.0833333358168602:
                                    return 8
                                else:
                                    if sample.get('sttl', 0) <= 0.6196078434586525:
                                        return 3
                                    else:
                                        return 5
                            else:
                                if sample.get('smean', 0) <= 0.040878377854824066:
                                    if sample.get('sbytes', 0) <= 9.96116523310775e-06:
                                        if sample.get('service', 0) <= 0.4999999925494194:
                                            if sample.get('sttl', 0) <= 0.6196078434586525:
                                                if sample.get('sbytes', 0) <= 7.453459375028615e-06:
                                                    return 2
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('ct_src_ltm', 0) <= 0.08474576286971569:
                                                    if sample.get('service', 0) <= 0.0833333358168602:
                                                        if sample.get('dload', 0) <= 3.371524144313298e-05:
                                                            if sample.get('sload', 0) <= 0.029549479484558105:
                                                                if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                                    if sample.get('sbytes', 0) <= 7.105167014742619e-06:
                                                                        return 3
                                                                    else:
                                                                        return 8
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('sbytes', 0) <= 8.21970297693042e-06:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                        return 2
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                        else:
                                                            return 2
                                                    else:
                                                        if sample.get('sbytes', 0) <= 6.617557346544345e-06:
                                                            if sample.get('smean', 0) <= 0.023310811258852482:
                                                                return 3
                                                            else:
                                                                return 5
                                                        else:
                                                            return 3
                                                else:
                                                    return 3
                                        else:
                                            if sample.get('sload', 0) <= 0.01040242938324809:
                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                    if sample.get('smean', 0) <= 0.027702702209353447:
                                                        return 3
                                                    else:
                                                        return 2
                                                else:
                                                    return 3
                                            else:
                                                return 7
                                    else:
                                        if sample.get('dmean', 0) <= 0.029666665941476822:
                                            if sample.get('proto', 0) <= 0.7307692170143127:
                                                return 2
                                            else:
                                                if sample.get('tcprtt', 0) <= 0.06823469698429108:
                                                    if sample.get('tcprtt', 0) <= 0.05502379313111305:
                                                        if sample.get('djit', 0) <= 0.000974191672867164:
                                                            if sample.get('ct_srv_dst', 0) <= 0.07377048954367638:
                                                                return 7
                                                            else:
                                                                if sample.get('dload', 0) <= 0.00018056356930173934:
                                                                    return 7
                                                                else:
                                                                    return 3
                                                        else:
                                                            if sample.get('dur', 0) <= 0.031952789053320885:
                                                                return 3
                                                            else:
                                                                return 7
                                                    else:
                                                        if sample.get('sbytes', 0) <= 3.740661486517638e-05:
                                                            return 3
                                                        else:
                                                            if sample.get('djit', 0) <= 0.001036371337249875:
                                                                if sample.get('synack', 0) <= 0.025028914213180542:
                                                                    return 3
                                                                else:
                                                                    return 7
                                                            else:
                                                                if sample.get('dtcpb', 0) <= 0.37950242310762405:
                                                                    return 3
                                                                else:
                                                                    return 8
                                                else:
                                                    if sample.get('sbytes', 0) <= 3.740661486517638e-05:
                                                        return 3
                                                    else:
                                                        if sample.get('stcpb', 0) <= 0.07398353889584541:
                                                            return 3
                                                        else:
                                                            return 7
                                        else:
                                            if sample.get('dmean', 0) <= 0.03633333183825016:
                                                if sample.get('dinpkt', 0) <= 0.0020637144334614277:
                                                    if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                                        return 3
                                                    else:
                                                        if sample.get('dloss', 0) <= 0.0002723806173889898:
                                                            return 4
                                                        else:
                                                            return 3
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('sload', 0) <= 9.864396588454838e-06:
                                                    return 3
                                                else:
                                                    return 3
                                else:
                                    if sample.get('proto', 0) <= 0.857692301273346:
                                        if sample.get('ct_src_dport_ltm', 0) <= 0.017241379246115685:
                                            if sample.get('smean', 0) <= 0.10337837785482407:
                                                if sample.get('dload', 0) <= 0.0003298115698271431:
                                                    if sample.get('sinpkt', 0) <= 0.004454227746464312:
                                                        return 3
                                                    else:
                                                        return 2
                                                else:
                                                    return 3
                                            else:
                                                return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                            if sample.get('service', 0) <= 0.0833333358168602:
                                                if sample.get('sttl', 0) <= 0.6196078434586525:
                                                    return 3
                                                else:
                                                    if sample.get('smean', 0) <= 0.10608107969164848:
                                                        if sample.get('proto', 0) <= 0.9038461446762085:
                                                            if sample.get('ct_src_ltm', 0) <= 0.17796609923243523:
                                                                if sample.get('sbytes', 0) <= 1.0379116247349884e-05:
                                                                    return 3
                                                                else:
                                                                    if sample.get('state', 0) <= 0.5000000149011612:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('sload', 0) <= 0.027131465263664722:
                                                                            if sample.get('smean', 0) <= 0.08479729667305946:
                                                                                return 8
                                                                            else:
                                                                                if sample.get('smean', 0) <= 0.08885135129094124:
                                                                                    return 3
                                                                                else:
                                                                                    return 8
                                                                        else:
                                                                            if sample.get('sload', 0) <= 0.029950222000479698:
                                                                                if sample.get('rate', 0) <= 0.1458333283662796:
                                                                                    return 6
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('sbytes', 0) <= 1.985267226700671e-05:
                                                                                    if sample.get('sload', 0) <= 0.04157175309956074:
                                                                                        return 3
                                                                                    else:
                                                                                        return 8
                                                                                else:
                                                                                    if sample.get('sload', 0) <= 0.09415337443351746:
                                                                                        return 3
                                                                                    else:
                                                                                        return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                                            if sample.get('ct_srv_src', 0) <= 0.09677419066429138:
                                                                if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                    if sample.get('smean', 0) <= 0.11047297343611717:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('sload', 0) <= 0.05270816944539547:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                else:
                                                                    return 6
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                if sample.get('rate', 0) <= 0.09166666492819786:
                                                                    return 3
                                                                else:
                                                                    return 6
                                                            else:
                                                                return 3
                                            else:
                                                return 3
                                        else:
                                            if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                return 3
                                            else:
                                                if sample.get('ct_srv_src', 0) <= 0.12096773833036423:
                                                    if sample.get('rate', 0) <= 0.1547618955373764:
                                                        if sample.get('smean', 0) <= 0.04493243247270584:
                                                            return 3
                                                        else:
                                                            return 4
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                    else:
                        if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                            if sample.get('sjit', 0) <= 0.002995978924445808:
                                if sample.get('sload', 0) <= 4.9708291044225916e-05:
                                    if sample.get('dur', 0) <= 0.009938576724380255:
                                        if sample.get('sbytes', 0) <= 6.944952474441379e-05:
                                            if sample.get('dmean', 0) <= 0.03033333271741867:
                                                if sample.get('smean', 0) <= 0.034121621400117874:
                                                    if sample.get('service', 0) <= 0.0833333358168602:
                                                        if sample.get('smean', 0) <= 0.023986486718058586:
                                                            if sample.get('sbytes', 0) <= 3.92177353205625e-05:
                                                                if sample.get('sbytes', 0) <= 3.9078417103155516e-05:
                                                                    if sample.get('tcprtt', 0) <= 0.013908409513533115:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.20234262198209763:
                                                                            return 6
                                                                        else:
                                                                            return 8
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('rate', 0) <= 2.6987397177435923e-05:
                                                                    return 3
                                                                else:
                                                                    if sample.get('sloss', 0) <= 0.0005640157760353759:
                                                                        return 6
                                                                    else:
                                                                        return 3
                                                        else:
                                                            if sample.get('djit', 0) <= 0.00028282770654186606:
                                                                if sample.get('sloss', 0) <= 0.0004700131685240194:
                                                                    if sample.get('ackdat', 0) <= 0.0036279635969549417:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.4418461173772812:
                                                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                if sample.get('smean', 0) <= 0.029391891323029995:
                                                                                    return 3
                                                                                else:
                                                                                    return 8
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('djit', 0) <= 0.00011547057511052117:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('tcprtt', 0) <= 0.0365178007632494:
                                                                                    return 8
                                                                                else:
                                                                                    if sample.get('dtcpb', 0) <= 0.3498324044048786:
                                                                                        return 6
                                                                                    else:
                                                                                        return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('sbytes', 0) <= 4.722846279037185e-05:
                                                            if sample.get('sbytes', 0) <= 3.7824565879418515e-05:
                                                                return 3
                                                            else:
                                                                if sample.get('dload', 0) <= 0.0002371053269598633:
                                                                    return 3
                                                                else:
                                                                    if sample.get('ackdat', 0) <= 0.015616069082170725:
                                                                        return 7
                                                                    else:
                                                                        return 3
                                                        else:
                                                            return 7
                                                else:
                                                    if sample.get('sbytes', 0) <= 5.872211477253586e-05:
                                                        if sample.get('stcpb', 0) <= 0.16847433149814606:
                                                            if sample.get('rate', 0) <= 7.81332746555563e-05:
                                                                if sample.get('dtcpb', 0) <= 0.8649272322654724:
                                                                    if sample.get('dtcpb', 0) <= 0.07180074788630009:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('tcprtt', 0) <= 0.04205808602273464:
                                                                            if sample.get('sload', 0) <= 5.721891056964523e-06:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('synack', 0) <= 0.0025393672985956073:
                                                                    return 7
                                                                else:
                                                                    return 3
                                                        else:
                                                            if sample.get('stcpb', 0) <= 0.9228156208992004:
                                                                if sample.get('service', 0) <= 0.2083333283662796:
                                                                    if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                        if sample.get('ackdat', 0) <= 0.009392825420945883:
                                                                            if sample.get('sjit', 0) <= 0.0020817500771954656:
                                                                                if sample.get('dtcpb', 0) <= 0.6550343036651611:
                                                                                    if sample.get('smean', 0) <= 0.037837838754057884:
                                                                                        return 3
                                                                                    else:
                                                                                        return 7
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('rate', 0) <= 3.4631792004802264e-05:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('synack', 0) <= 0.0020737339509651065:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('synack', 0) <= 0.015452827792614698:
                                                                                        if sample.get('ackdat', 0) <= 0.018636953085660934:
                                                                                            return 7
                                                                                        else:
                                                                                            if sample.get('tcprtt', 0) <= 0.022408291697502136:
                                                                                                return 3
                                                                                            else:
                                                                                                return 7
                                                                                    else:
                                                                                        if sample.get('smean', 0) <= 0.03885135054588318:
                                                                                            return 7
                                                                                        else:
                                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('smean', 0) <= 0.03614864870905876:
                                                                        if sample.get('sbytes', 0) <= 5.126865653437562e-05:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('ct_flw_http_mthd', 0) <= 0.15625:
                                                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        if sample.get('dtcpb', 0) <= 0.10546782612800598:
                                                                            if sample.get('sinpkt', 0) <= 0.0003760947729460895:
                                                                                return 2
                                                                            else:
                                                                                if sample.get('synack', 0) <= 0.017533999867737293:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('sbytes', 0) <= 5.370670260163024e-05:
                                                                                        return 3
                                                                                    else:
                                                                                        return 7
                                                                        else:
                                                                            if sample.get('dload', 0) <= 0.00045175837294664234:
                                                                                if sample.get('synack', 0) <= 0.005202386993914843:
                                                                                    if sample.get('tcprtt', 0) <= 0.007677029585465789:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('smean', 0) <= 0.03682432509958744:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('dinpkt', 0) <= 0.0006117364682722837:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('synack', 0) <= 0.0019046803936362267:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.22033897787332535:
                                                                                        if sample.get('tcprtt', 0) <= 0.01681711059063673:
                                                                                            if sample.get('ackdat', 0) <= 0.006425888277590275:
                                                                                                return 7
                                                                                            else:
                                                                                                if sample.get('dtcpb', 0) <= 0.6889268755912781:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 3
                                                                                        else:
                                                                                            if sample.get('dtcpb', 0) <= 0.9805210828781128:
                                                                                                if sample.get('ct_srv_src', 0) <= 0.12903225421905518:
                                                                                                    if sample.get('djit', 0) <= 0.00031154925818555057:
                                                                                                        if sample.get('dur', 0) <= 0.0061636262107640505:
                                                                                                            if sample.get('sjit', 0) <= 0.0011001926031894982:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                if sample.get('dtcpb', 0) <= 0.22669506818056107:
                                                                                                                    return 3
                                                                                                                else:
                                                                                                                    return 3
                                                                                                        else:
                                                                                                            return 7
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    return 3
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                if sample.get('synack', 0) <= 0.0018529261578805745:
                                                                                    return 3
                                                                                else:
                                                                                    return 7
                                                            else:
                                                                if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.18644067272543907:
                                                                        if sample.get('ackdat', 0) <= 0.03661714866757393:
                                                                            if sample.get('djit', 0) <= 0.00012671700824284926:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 7
                                                    else:
                                                        if sample.get('dinpkt', 0) <= 0.0002290348882922899:
                                                            return 3
                                                        else:
                                                            if sample.get('stcpb', 0) <= 0.6488266289234161:
                                                                if sample.get('dtcpb', 0) <= 0.8852619230747223:
                                                                    if sample.get('sjit', 0) <= 0.0006516728608403355:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dur', 0) <= 0.009670051746070385:
                                                                            if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                                if sample.get('sbytes', 0) <= 6.917088830959983e-05:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.2288135588169098:
                                                                                        if sample.get('dtcpb', 0) <= 0.020057679153978825:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('ct_dst_ltm', 0) <= 0.09482758305966854:
                                                                                                if sample.get('ct_src_ltm', 0) <= 0.05932203307747841:
                                                                                                    if sample.get('sbytes', 0) <= 6.74294242344331e-05:
                                                                                                        if sample.get('stcpb', 0) <= 0.28554193675518036:
                                                                                                            if sample.get('sbytes', 0) <= 6.025460061209742e-05:
                                                                                                                if sample.get('dtcpb', 0) <= 0.6094841361045837:
                                                                                                                    return 7
                                                                                                                else:
                                                                                                                    return 3
                                                                                                            else:
                                                                                                                if sample.get('stcpb', 0) <= 0.2402235046029091:
                                                                                                                    if sample.get('sbytes', 0) <= 6.35982105450239e-05:
                                                                                                                        return 3
                                                                                                                    else:
                                                                                                                        if sample.get('dmean', 0) <= 0.029666665941476822:
                                                                                                                            return 7
                                                                                                                        else:
                                                                                                                            return 3
                                                                                                                else:
                                                                                                                    if sample.get('dtcpb', 0) <= 0.1124589703977108:
                                                                                                                        return 3
                                                                                                                    else:
                                                                                                                        return 7
                                                                                                        else:
                                                                                                            if sample.get('stcpb', 0) <= 0.6382358372211456:
                                                                                                                if sample.get('synack', 0) <= 0.0025398321449756622:
                                                                                                                    if sample.get('dinpkt', 0) <= 0.0006782423879485577:
                                                                                                                        if sample.get('synack', 0) <= 0.0025071371346712112:
                                                                                                                            return 3
                                                                                                                        else:
                                                                                                                            return 3
                                                                                                                    else:
                                                                                                                        return 3
                                                                                                                else:
                                                                                                                    return 3
                                                                                                            else:
                                                                                                                return 3
                                                                                                    else:
                                                                                                        if sample.get('djit', 0) <= 0.00013237819075584412:
                                                                                                            if sample.get('synack', 0) <= 0.006696442374959588:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                if sample.get('dload', 0) <= 0.000427855018642731:
                                                                                                                    return 3
                                                                                                                else:
                                                                                                                    return 3
                                                                                                        else:
                                                                                                            if sample.get('dtcpb', 0) <= 0.08373968303203583:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                if sample.get('stcpb', 0) <= 0.4947456270456314:
                                                                                                                    return 3
                                                                                                                else:
                                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('ct_src_ltm', 0) <= 0.0762711875140667:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        if sample.get('sbytes', 0) <= 6.032426063029561e-05:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 3
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('synack', 0) <= 0.0020402641966938972:
                                                                                    if sample.get('dur', 0) <= 0.0030564755434170365:
                                                                                        return 3
                                                                                    else:
                                                                                        return 7
                                                                                else:
                                                                                    if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                                                                        if sample.get('stcpb', 0) <= 0.2517833560705185:
                                                                                            if sample.get('sbytes', 0) <= 6.868327909614891e-05:
                                                                                                if sample.get('stcpb', 0) <= 0.011763409711420536:
                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('dinpkt', 0) <= 0.0006469309446401894:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            if sample.get('dtcpb', 0) <= 0.6780804693698883:
                                                                                                if sample.get('synack', 0) <= 0.011627351865172386:
                                                                                                    if sample.get('synack', 0) <= 0.002587402705103159:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 7
                                                                                                else:
                                                                                                    if sample.get('stcpb', 0) <= 0.3699796497821808:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        if sample.get('sjit', 0) <= 0.0006790480692870915:
                                                                                            return 3
                                                                                        else:
                                                                                            return 3
                                                                        else:
                                                                            if sample.get('sinpkt', 0) <= 0.0010285176103934646:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                else:
                                                                    if sample.get('djit', 0) <= 0.00018021151481661946:
                                                                        if sample.get('service', 0) <= 0.2083333283662796:
                                                                            return 3
                                                                        else:
                                                                            return 7
                                                                    else:
                                                                        if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                            if sample.get('stcpb', 0) <= 0.09087809268385172:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('dtcpb', 0) <= 0.9793221056461334:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 3
                                                            else:
                                                                if sample.get('stcpb', 0) <= 0.694273829460144:
                                                                    return 3
                                                                else:
                                                                    if sample.get('service', 0) <= 0.2083333283662796:
                                                                        if sample.get('smean', 0) <= 0.042905405163764954:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('synack', 0) <= 0.00247630150988698:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('synack', 0) <= 0.015230780467391014:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.9048643410205841:
                                                                            if sample.get('djit', 0) <= 0.0001265050668735057:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('tcprtt', 0) <= 0.017413086257874966:
                                                                                    if sample.get('sload', 0) <= 7.02327088220045e-06:
                                                                                        if sample.get('synack', 0) <= 0.0020467721624299884:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('dtcpb', 0) <= 0.33339227735996246:
                                                                                                return 3
                                                                                            else:
                                                                                                return 7
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    if sample.get('ackdat', 0) <= 0.01934236753731966:
                                                                                        if sample.get('ct_dst_ltm', 0) <= 0.09482758771628141:
                                                                                            return 3
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('stcpb', 0) <= 0.8194933533668518:
                                                                                            if sample.get('stcpb', 0) <= 0.7959468066692352:
                                                                                                if sample.get('stcpb', 0) <= 0.7388890087604523:
                                                                                                    if sample.get('ct_src_ltm', 0) <= 0.042372881434857845:
                                                                                                        return 2
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    return 3
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            return 3
                                                                        else:
                                                                            if sample.get('dtcpb', 0) <= 0.19704259186983109:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('rate', 0) <= 3.244331855967175e-05:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('dinpkt', 0) <= 0.000583689718041569:
                                                                                        return 3
                                                                                    else:
                                                                                        return 7
                                            else:
                                                if sample.get('dur', 0) <= 0.008833010215312243:
                                                    if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                                        if sample.get('dmean', 0) <= 0.08966666832566261:
                                                            if sample.get('dload', 0) <= 0.000628706329734996:
                                                                if sample.get('synack', 0) <= 0.002343042055144906:
                                                                    return 2
                                                                else:
                                                                    if sample.get('tcprtt', 0) <= 0.010011082049459219:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                if sample.get('sbytes', 0) <= 5.39853372174548e-05:
                                                                    if sample.get('sbytes', 0) <= 4.792504660144914e-05:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('response_body_len', 0) <= 2.288818359375e-05:
                                                                            if sample.get('synack', 0) <= 0.003970047109760344:
                                                                                if sample.get('synack', 0) <= 0.0022558036725968122:
                                                                                    return 2
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 2
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('dtcpb', 0) <= 0.15187383443117142:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dtcpb', 0) <= 0.662121057510376:
                                                                            if sample.get('djit', 0) <= 0.00010718409976107068:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('sbytes', 0) <= 6.366786692524329e-05:
                                                                                    if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                                                                        return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            if sample.get('ackdat', 0) <= 0.015273775905370712:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                    return 2
                                                                                else:
                                                                                    return 3
                                                        else:
                                                            if sample.get('ackdat', 0) <= 0.025948194786906242:
                                                                if sample.get('smean', 0) <= 0.03749999962747097:
                                                                    if sample.get('sinpkt', 0) <= 0.0009630710992496461:
                                                                        if sample.get('tcprtt', 0) <= 0.007017989177256823:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('ackdat', 0) <= 0.01235737232491374:
                                                                                if sample.get('ackdat', 0) <= 0.009451723657548428:
                                                                                    return 3
                                                                                else:
                                                                                    return 2
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('sinpkt', 0) <= 0.0004208151076454669:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('response_body_len', 0) <= 3.62396240234375e-05:
                                                                            if sample.get('service', 0) <= 0.2083333283662796:
                                                                                if sample.get('stcpb', 0) <= 0.8265427649021149:
                                                                                    if sample.get('sbytes', 0) <= 5.454260463011451e-05:
                                                                                        if sample.get('tcprtt', 0) <= 0.012315564788877964:
                                                                                            return 3
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('rate', 0) <= 7.050656859064475e-05:
                                                                                            if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                                                return 3
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    if sample.get('sjit', 0) <= 0.001656693872064352:
                                                                                        return 2
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                if sample.get('ct_dst_ltm', 0) <= 0.0258620698004961:
                                                                                    return 3
                                                                                else:
                                                                                    return 2
                                                                        else:
                                                                            if sample.get('synack', 0) <= 0.0021103029139339924:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('sjit', 0) <= 0.0010062323417514563:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.06779661029577255:
                                                                                        if sample.get('synack', 0) <= 0.0021639165934175253:
                                                                                            return 3
                                                                                        else:
                                                                                            return 2
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    if sample.get('dtcpb', 0) <= 0.18006784468889236:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('dtcpb', 0) <= 0.27196721732616425:
                                                                                            return 2
                                                                                        else:
                                                                                            if sample.get('dinpkt', 0) <= 0.0008937608217820525:
                                                                                                if sample.get('dur', 0) <= 0.005590609274804592:
                                                                                                    if sample.get('synack', 0) <= 0.0031063397182151675:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 2
                                                                                                else:
                                                                                                    if sample.get('stcpb', 0) <= 0.09629220521310344:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                            else:
                                                                                                return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('sinpkt', 0) <= 0.00046229946019593626:
                                                            return 6
                                                        else:
                                                            if sample.get('dtcpb', 0) <= 0.9680018424987793:
                                                                if sample.get('sloss', 0) <= 0.0008460236713290215:
                                                                    if sample.get('sbytes', 0) <= 4.2491683416301385e-05:
                                                                        if sample.get('dur', 0) <= 0.006758142961189151:
                                                                            return 3
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                else:
                                                    if sample.get('dload', 0) <= 0.0007129094447009265:
                                                        if sample.get('dload', 0) <= 0.0003048516809940338:
                                                            return 3
                                                        else:
                                                            if sample.get('sjit', 0) <= 0.0023379025515168905:
                                                                if sample.get('dloss', 0) <= 0.00063555475207977:
                                                                    if sample.get('stcpb', 0) <= 0.10152675211429596:
                                                                        return 3
                                                                    else:
                                                                        return 6
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('service', 0) <= 0.2083333283662796:
                                                                    return 5
                                                                else:
                                                                    return 2
                                                    else:
                                                        if sample.get('stcpb', 0) <= 0.4577937722206116:
                                                            return 3
                                                        else:
                                                            return 3
                                        else:
                                            if sample.get('dur', 0) <= 0.0022283587604761124:
                                                if sample.get('dmean', 0) <= 0.10566666722297668:
                                                    return 3
                                                else:
                                                    return 5
                                            else:
                                                if sample.get('dur', 0) <= 0.006761176278814673:
                                                    if sample.get('sttl', 0) <= 0.6196078434586525:
                                                        if sample.get('dinpkt', 0) <= 0.0006630957068409771:
                                                            if sample.get('response_body_len', 0) <= 9.536743306171047e-08:
                                                                if sample.get('dtcpb', 0) <= 0.1064561977982521:
                                                                    return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('dmean', 0) <= 0.0793333351612091:
                                                                    return 3
                                                                else:
                                                                    if sample.get('sinpkt', 0) <= 0.000424495039624162:
                                                                        return 2
                                                                    else:
                                                                        return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('sbytes', 0) <= 6.958884114283137e-05:
                                                            return 3
                                                        else:
                                                            if sample.get('rate', 0) <= 8.5170006059343e-05:
                                                                if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                    if sample.get('djit', 0) <= 0.00012891453661723062:
                                                                        if sample.get('smean', 0) <= 0.05709459446370602:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('djit', 0) <= 0.00012852576764998958:
                                                                                if sample.get('rate', 0) <= 8.410745795117691e-05:
                                                                                    if sample.get('dload', 0) <= 0.00041958820656873286:
                                                                                        if sample.get('sinpkt', 0) <= 0.00039150702650658786:
                                                                                            if sample.get('sjit', 0) <= 0.0007786669593770057:
                                                                                                return 3
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            if sample.get('dpkts', 0) <= 0.000635324016911909:
                                                                                                return 3
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        if sample.get('djit', 0) <= 9.019106073537841e-05:
                                                                                            if sample.get('stcpb', 0) <= 0.6713951230049133:
                                                                                                return 3
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            if sample.get('sload', 0) <= 2.3698898985458072e-05:
                                                                                                if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 3
                                                                                            else:
                                                                                                return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('rate', 0) <= 7.310524233616889e-05:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                    else:
                                                                        if sample.get('smean', 0) <= 0.049662161618471146:
                                                                            return 2
                                                                        else:
                                                                            if sample.get('rate', 0) <= 3.9589242078363895e-05:
                                                                                if sample.get('dload', 0) <= 0.00022486821399070323:
                                                                                    if sample.get('dtcpb', 0) <= 0.7206237018108368:
                                                                                        return 3
                                                                                    else:
                                                                                        return 8
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('dtcpb', 0) <= 0.35660769045352936:
                                                                                    if sample.get('stcpb', 0) <= 0.05322679877281189:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('dinpkt', 0) <= 0.0005817845812998712:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('ackdat', 0) <= 0.02491090167313814:
                                                                                                if sample.get('dtcpb', 0) <= 0.33433155715465546:
                                                                                                    if sample.get('dtcpb', 0) <= 0.09500706195831299:
                                                                                                        if sample.get('dtcpb', 0) <= 0.06519503332674503:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            if sample.get('dload', 0) <= 0.0003635755128925666:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                return 3
                                                                                                    else:
                                                                                                        if sample.get('synack', 0) <= 0.0021482662996277213:
                                                                                                            if sample.get('synack', 0) <= 0.0020766780944541097:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                return 3
                                                                                                        else:
                                                                                                            return 3
                                                                                                else:
                                                                                                    return 9
                                                                                            else:
                                                                                                return 3
                                                                                else:
                                                                                    if sample.get('sinpkt', 0) <= 0.0006629766430705786:
                                                                                        if sample.get('tcprtt', 0) <= 0.006421490339562297:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('dtcpb', 0) <= 0.9695093035697937:
                                                                                                if sample.get('dinpkt', 0) <= 0.000625650747679174:
                                                                                                    if sample.get('dinpkt', 0) <= 0.000623366329818964:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    return 3
                                                                                            else:
                                                                                                if sample.get('dtcpb', 0) <= 0.9737324416637421:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 3
                                                                                    else:
                                                                                        if sample.get('stcpb', 0) <= 0.7659524083137512:
                                                                                            return 3
                                                                                        else:
                                                                                            return 2
                                                                else:
                                                                    if sample.get('dtcpb', 0) <= 0.3769131153821945:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dinpkt', 0) <= 0.0005687795637641102:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                            else:
                                                                if sample.get('rate', 0) <= 8.562070797779597e-05:
                                                                    return 3
                                                                else:
                                                                    if sample.get('dtcpb', 0) <= 0.40771879255771637:
                                                                        if sample.get('dtcpb', 0) <= 0.17509503662586212:
                                                                            return 3
                                                                        else:
                                                                            return 2
                                                                    else:
                                                                        return 3
                                                else:
                                                    if sample.get('ct_src_ltm', 0) <= 0.3474576324224472:
                                                        if sample.get('dmean', 0) <= 0.08899999782443047:
                                                            if sample.get('dinpkt', 0) <= 0.0007078871130943298:
                                                                if sample.get('sinpkt', 0) <= 0.0005568258347921073:
                                                                    return 3
                                                                else:
                                                                    return 2
                                                            else:
                                                                if sample.get('sload', 0) <= 4.673258899856592e-06:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.1355932168662548:
                                                                        if sample.get('ct_dst_ltm', 0) <= 0.09482758305966854:
                                                                            if sample.get('ackdat', 0) <= 0.042335232719779015:
                                                                                if sample.get('dtcpb', 0) <= 0.37499330937862396:
                                                                                    if sample.get('synack', 0) <= 0.02542373351752758:
                                                                                        if sample.get('dtcpb', 0) <= 0.34133554995059967:
                                                                                            if sample.get('rate', 0) <= 3.558391108526848e-05:
                                                                                                return 3
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('rate', 0) <= 3.1905967261991464e-05:
                                                                                            return 9
                                                                                        else:
                                                                                            return 7
                                                                                else:
                                                                                    if sample.get('smean', 0) <= 0.08513513207435608:
                                                                                        if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                                            if sample.get('stcpb', 0) <= 0.9380864500999451:
                                                                                                return 3
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            if sample.get('smean', 0) <= 0.0635135117918253:
                                                                                                return 3
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('sbytes', 0) <= 0.00011089632971561514:
                                                                        return 5
                                                                    else:
                                                                        if sample.get('sbytes', 0) <= 0.00012643017544178292:
                                                                            if sample.get('ackdat', 0) <= 0.02296111173927784:
                                                                                if sample.get('smean', 0) <= 0.10472973063588142:
                                                                                    return 2
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('dtcpb', 0) <= 0.27305230498313904:
                                                                                if sample.get('dload', 0) <= 0.0002934971998911351:
                                                                                    if sample.get('ackdat', 0) <= 0.030316910706460476:
                                                                                        if sample.get('dtcpb', 0) <= 0.1704646721482277:
                                                                                            if sample.get('dinpkt', 0) <= 0.0016886349767446518:
                                                                                                return 3
                                                                                            else:
                                                                                                return 2
                                                                                        else:
                                                                                            if sample.get('sjit', 0) <= 0.0022662259871140122:
                                                                                                return 5
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('rate', 0) <= 2.842742469510995e-05:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('ackdat', 0) <= 0.02236803248524666:
                                                                                        if sample.get('sload', 0) <= 2.0933499399689026e-05:
                                                                                            return 3
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                                            if sample.get('stcpb', 0) <= 0.3426797688007355:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('sload', 0) <= 2.994694841618184e-05:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 3
                                                                                        else:
                                                                                            return 3
                                                        else:
                                                            return 5
                                                    else:
                                                        return 3
                                    else:
                                        if sample.get('proto', 0) <= 0.7923076748847961:
                                            if sample.get('spkts', 0) <= 0.001690934703219682:
                                                return 2
                                            else:
                                                if sample.get('sbytes', 0) <= 0.0009568291716277599:
                                                    if sample.get('state', 0) <= 0.75:
                                                        if sample.get('dur', 0) <= 0.5067923069000244:
                                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                if sample.get('smean', 0) <= 0.041891892440617085:
                                                                    if sample.get('dur', 0) <= 0.4188452959060669:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dur', 0) <= 0.48612603545188904:
                                                                            return 2
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('sinpkt', 0) <= 0.0007582882244605571:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('sjit', 0) <= 0.00015147735393838957:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('dur', 0) <= 0.2098952755331993:
                                                                                return 2
                                                                            else:
                                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 2
                                                    else:
                                                        return 2
                                                else:
                                                    return 3
                                        else:
                                            if sample.get('service', 0) <= 0.3333333283662796:
                                                if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                                    if sample.get('dloss', 0) <= 9.079353912966326e-05:
                                                        return 3
                                                    else:
                                                        if sample.get('dbytes', 0) <= 5.532991963264067e-05:
                                                            if sample.get('dbytes', 0) <= 3.786449451581575e-05:
                                                                if sample.get('synack', 0) <= 0.021711373701691628:
                                                                    if sample.get('sbytes', 0) <= 5.085070552013349e-05:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dload', 0) <= 0.00013695310190087184:
                                                                            if sample.get('sbytes', 0) <= 6.561830559803639e-05:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('sload', 0) <= 3.7276203102010186e-06:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('stcpb', 0) <= 0.3050740361213684:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('rate', 0) <= 2.5340334104839712e-05:
                                                                                            return 5
                                                                                        else:
                                                                                            return 3
                                                                        else:
                                                                            if sample.get('dtcpb', 0) <= 0.8529148995876312:
                                                                                if sample.get('djit', 0) <= 0.0002591900702100247:
                                                                                    return 5
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                else:
                                                                    if sample.get('tcprtt', 0) <= 0.03568134270608425:
                                                                        if sample.get('dur', 0) <= 0.011161643546074629:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('ackdat', 0) <= 0.018480745144188404:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.8926888406276703:
                                                                            if sample.get('dmean', 0) <= 0.03033333271741867:
                                                                                if sample.get('ct_srv_src', 0) <= 0.08870967477560043:
                                                                                    if sample.get('dur', 0) <= 0.01039827661588788:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('sjit', 0) <= 0.0029181934660300612:
                                                                                            if sample.get('dinpkt', 0) <= 0.0023170344065874815:
                                                                                                return 6
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('djit', 0) <= 0.0002963842125609517:
                                                                                    return 5
                                                                                else:
                                                                                    if sample.get('dmean', 0) <= 0.03499999921768904:
                                                                                        return 3
                                                                                    else:
                                                                                        return 3
                                                                        else:
                                                                            return 3
                                                            else:
                                                                if sample.get('tcprtt', 0) <= 0.024462084285914898:
                                                                    if sample.get('smean', 0) <= 0.027364864945411682:
                                                                        if sample.get('dbytes', 0) <= 4.7143001211225055e-05:
                                                                            if sample.get('sload', 0) <= 1.4504602745546435e-06:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('dmean', 0) <= 0.060999998822808266:
                                                                        if sample.get('dtcpb', 0) <= 0.9263525009155273:
                                                                            if sample.get('dloss', 0) <= 0.0008171418448910117:
                                                                                if sample.get('dbytes', 0) <= 4.345888737589121e-05:
                                                                                    if sample.get('sloss', 0) <= 0.000658018427202478:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('dtcpb', 0) <= 0.4979131668806076:
                                                                                            return 6
                                                                                        else:
                                                                                            if sample.get('dtcpb', 0) <= 0.5475331246852875:
                                                                                                return 3
                                                                                            else:
                                                                                                return 6
                                                                                else:
                                                                                    return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                        else:
                                                            if sample.get('ackdat', 0) <= 0.01985042914748192:
                                                                if sample.get('synack', 0) <= 0.0068056839518249035:
                                                                    return 3
                                                                else:
                                                                    if sample.get('rate', 0) <= 3.113532511633821e-05:
                                                                        if sample.get('sjit', 0) <= 0.0024688933044672012:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                if sample.get('smean', 0) <= 0.5547297149896622:
                                                                    if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                                                        if sample.get('dinpkt', 0) <= 0.001511504000518471:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('dbytes', 0) <= 7.040749187581241e-05:
                                                                                return 5
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                else:
                                                    if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                        if sample.get('synack', 0) <= 0.007482363376766443:
                                                            return 3
                                                        else:
                                                            if sample.get('dload', 0) <= 0.00011572059884201735:
                                                                if sample.get('ackdat', 0) <= 0.044618950225412846:
                                                                    return 8
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('dtcpb', 0) <= 0.12967970594763756:
                                                                    return 3
                                                                else:
                                                                    if sample.get('sjit', 0) <= 0.00289229245390743:
                                                                        return 6
                                                                    else:
                                                                        return 3
                                                    else:
                                                        if sample.get('dur', 0) <= 0.011484027374535799:
                                                            if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                                if sample.get('dur', 0) <= 0.011360251810401678:
                                                                    if sample.get('smean', 0) <= 0.023310811258852482:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('synack', 0) <= 0.02380773052573204:
                                                                            return 3
                                                                        else:
                                                                            return 6
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('dinpkt', 0) <= 0.0012260316871106625:
                                                                if sample.get('sbytes', 0) <= 9.362102355225943e-05:
                                                                    return 5
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('sjit', 0) <= 4.8060930566862226e-05:
                                                                    return 3
                                                                else:
                                                                    if sample.get('dmean', 0) <= 0.0663333348929882:
                                                                        if sample.get('ackdat', 0) <= 0.021289083175361156:
                                                                            if sample.get('dtcpb', 0) <= 0.4118533134460449:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('sjit', 0) <= 0.002834881073795259:
                                                                                    return 4
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        return 3
                                            else:
                                                if sample.get('sbytes', 0) <= 5.858279655512888e-05:
                                                    if sample.get('dloss', 0) <= 0.0002723806173889898:
                                                        if sample.get('tcprtt', 0) <= 0.029724856838583946:
                                                            return 3
                                                        else:
                                                            if sample.get('sload', 0) <= 1.194445758301299e-06:
                                                                return 4
                                                            else:
                                                                if sample.get('ct_dst_ltm', 0) <= 0.034482759423553944:
                                                                    if sample.get('tcprtt', 0) <= 0.06127597764134407:
                                                                        if sample.get('djit', 0) <= 0.0002628125948831439:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('sbytes', 0) <= 5.684133429895155e-05:
                                                                                if sample.get('stcpb', 0) <= 0.9047761559486389:
                                                                                    return 7
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                    else:
                                                        if sample.get('dbytes', 0) <= 8.81458145158831e-05:
                                                            if sample.get('sbytes', 0) <= 5.558748307521455e-05:
                                                                if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                    if sample.get('sinpkt', 0) <= 0.0011339195189066231:
                                                                        return 3
                                                                    else:
                                                                        return 2
                                                                else:
                                                                    return 6
                                                            else:
                                                                if sample.get('dload', 0) <= 0.0006676792690996081:
                                                                    if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                                                        return 3
                                                                    else:
                                                                        return 6
                                                                else:
                                                                    return 3
                                                        else:
                                                            if sample.get('sbytes', 0) <= 5.46122628293233e-05:
                                                                return 3
                                                            else:
                                                                return 3
                                                else:
                                                    if sample.get('dpkts', 0) <= 0.001542929734569043:
                                                        if sample.get('sinpkt', 0) <= 0.001360055000986904:
                                                            if sample.get('dur', 0) <= 0.011933168862015009:
                                                                if sample.get('dur', 0) <= 0.011738177388906479:
                                                                    if sample.get('ct_dst_ltm', 0) <= 0.025862068869173527:
                                                                        if sample.get('stcpb', 0) <= 0.6904654502868652:
                                                                            if sample.get('stcpb', 0) <= 0.6206039488315582:
                                                                                if sample.get('ackdat', 0) <= 0.028801431879401207:
                                                                                    if sample.get('rate', 0) <= 2.975318147946382e-05:
                                                                                        if sample.get('sjit', 0) <= 0.0028348079649731517:
                                                                                            if sample.get('sbytes', 0) <= 8.28936135803815e-05:
                                                                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                                                    if sample.get('sjit', 0) <= 0.0026638148119673133:
                                                                                                        if sample.get('djit', 0) <= 0.00031619872606825083:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    return 3
                                                                                            else:
                                                                                                if sample.get('dur', 0) <= 0.010908727068454027:
                                                                                                    return 1
                                                                                                else:
                                                                                                    return 3
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('stcpb', 0) <= 0.5141516327857971:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('dtcpb', 0) <= 0.558520570397377:
                                                                                                return 2
                                                                                            else:
                                                                                                return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('dload', 0) <= 0.00012611199781531468:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('smean', 0) <= 0.07229729741811752:
                                                                                            return 4
                                                                                        else:
                                                                                            return 3
                                                                        else:
                                                                            if sample.get('sjit', 0) <= 0.0019192625768482685:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('stcpb', 0) <= 0.8776428699493408:
                                                                    if sample.get('tcprtt', 0) <= 0.06087639182806015:
                                                                        if sample.get('dinpkt', 0) <= 0.0025216075591742992:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('dinpkt', 0) <= 0.0012894979445263743:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('ct_flw_http_mthd', 0) <= 0.03125:
                                                                            return 3
                                                                        else:
                                                                            return 4
                                                        else:
                                                            if sample.get('djit', 0) <= 0.0003883305034833029:
                                                                if sample.get('dur', 0) <= 0.10464630275964737:
                                                                    return 5
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        return 3
                                else:
                                    if sample.get('dload', 0) <= 0.00022562802041647956:
                                        if sample.get('sbytes', 0) <= 0.0001985963826882653:
                                            if sample.get('sbytes', 0) <= 7.26538164599333e-05:
                                                if sample.get('smean', 0) <= 0.33479729294776917:
                                                    if sample.get('proto', 0) <= 0.8153845965862274:
                                                        return 3
                                                    else:
                                                        if sample.get('sbytes', 0) <= 6.150845365482382e-05:
                                                            if sample.get('sload', 0) <= 0.0480469036847353:
                                                                if sample.get('smean', 0) <= 0.18209458887577057:
                                                                    return 3
                                                                else:
                                                                    if sample.get('sttl', 0) <= 0.6196078434586525:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('sbytes', 0) <= 3.9496369936387055e-05:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('sbytes', 0) <= 4.8412657633889467e-05:
                                                                                return 8
                                                                            else:
                                                                                return 2
                                                            else:
                                                                if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                    if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                                                        if sample.get('sload', 0) <= 0.058550573885440826:
                                                                            if sample.get('sload', 0) <= 0.05570511147379875:
                                                                                if sample.get('sload', 0) <= 0.04996203072369099:
                                                                                    return 3
                                                                                else:
                                                                                    return 8
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                                if sample.get('smean', 0) <= 0.23547297716140747:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                        if sample.get('service', 0) <= 0.0416666679084301:
                                                                                            if sample.get('rate', 0) <= 0.1547618955373764:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('smean', 0) <= 0.20641891658306122:
                                                                                                    return 5
                                                                                                else:
                                                                                                    return 3
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 6
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 6
                                                        else:
                                                            return 3
                                                else:
                                                    if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                                        if sample.get('sload', 0) <= 0.06473531015217304:
                                                            return 3
                                                        else:
                                                            return 6
                                                    else:
                                                        if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                            if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                    if sample.get('sload', 0) <= 0.07711741328239441:
                                                                        return 3
                                                                    else:
                                                                        return 6
                                                                else:
                                                                    if sample.get('sload', 0) <= 0.07711741328239441:
                                                                        return 4
                                                                    else:
                                                                        if sample.get('sload', 0) <= 0.08494051545858383:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                                return 4
                                                                            else:
                                                                                return 3
                                                            else:
                                                                return 6
                                                        else:
                                                            if sample.get('rate', 0) <= 0.12142856791615486:
                                                                return 3
                                                            else:
                                                                if sample.get('rate', 0) <= 0.17142856866121292:
                                                                    return 3
                                                                else:
                                                                    return 6
                                            else:
                                                if sample.get('sbytes', 0) <= 0.00010337321145925671:
                                                    if sample.get('sload', 0) <= 0.07623279839754105:
                                                        return 3
                                                    else:
                                                        if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                                            if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                if sample.get('sbytes', 0) <= 8.163975871866569e-05:
                                                                    if sample.get('smean', 0) <= 0.36486485600471497:
                                                                        if sample.get('rate', 0) <= 0.22500000149011612:
                                                                            if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                if sample.get('sload', 0) <= 0.12781573459506035:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                            return 2
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('sbytes', 0) <= 0.0001001689197437372:
                                                                        if sample.get('sttl', 0) <= 0.6196078434586525:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('sbytes', 0) <= 8.958082980825566e-05:
                                                                                return 5
                                                                            else:
                                                                                if sample.get('sbytes', 0) <= 8.9929122623289e-05:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('sbytes', 0) <= 9.076502465177327e-05:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('sload', 0) <= 0.09684544056653976:
                                                                                            return 5
                                                                                        else:
                                                                                            if sample.get('sload', 0) <= 0.11735424771904945:
                                                                                                if sample.get('rate', 0) <= 0.09545454755425453:
                                                                                                    return 2
                                                                                                else:
                                                                                                    return 3
                                                                                            else:
                                                                                                if sample.get('smean', 0) <= 0.45236486196517944:
                                                                                                    return 5
                                                                                                else:
                                                                                                    return 3
                                                                    else:
                                                                        if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                            return 3
                                                                        else:
                                                                            return 8
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                                else:
                                                    if sample.get('smean', 0) <= 0.7260135114192963:
                                                        if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                if sample.get('sbytes', 0) <= 0.00011284677020739764:
                                                                    return 6
                                                                else:
                                                                    if sample.get('smean', 0) <= 0.6260135173797607:
                                                                        if sample.get('sload', 0) <= 0.3234623968601227:
                                                                            if sample.get('sload', 0) <= 0.10263710841536522:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('smean', 0) <= 0.7087837755680084:
                                                                            if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                                if sample.get('smean', 0) <= 0.6537162363529205:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('sbytes', 0) <= 0.00014384480164153501:
                                                                                        return 9
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                if sample.get('smean', 0) <= 0.6922297477722168:
                                                                                    return 6
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('sbytes', 0) <= 0.00017435522022424266:
                                                            if sample.get('sload', 0) <= 0.4246393293142319:
                                                                if sample.get('ct_srv_src', 0) <= 0.016129031777381897:
                                                                    if sample.get('sload', 0) <= 0.16899287700653076:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                if sample.get('sload', 0) <= 0.3371298164129257:
                                                                    if sample.get('sbytes', 0) <= 0.00018870487110689282:
                                                                        if sample.get('sload', 0) <= 0.22593434900045395:
                                                                            if sample.get('sbytes', 0) <= 0.00018173902935814112:
                                                                                return 2
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                        else:
                                            if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                                if sample.get('dpkts', 0) <= 0.0011798874475061893:
                                                    if sample.get('djit', 0) <= 0.0002682068006834015:
                                                        if sample.get('sload', 0) <= 7.438959801220335e-05:
                                                            if sample.get('sbytes', 0) <= 0.002523100469261408:
                                                                return 6
                                                            else:
                                                                return 3
                                                        else:
                                                            return 4
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('service', 0) <= 0.3333333283662796:
                                                        if sample.get('dtcpb', 0) <= 0.9416049718856812:
                                                            if sample.get('smean', 0) <= 0.8179054260253906:
                                                                if sample.get('synack', 0) <= 0.05351141840219498:
                                                                    if sample.get('smean', 0) <= 0.8030405640602112:
                                                                        if sample.get('dur', 0) <= 0.02005829568952322:
                                                                            if sample.get('sbytes', 0) <= 0.0049138497561216354:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('smean', 0) <= 0.7152027189731598:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                    else:
                                                                        if sample.get('smean', 0) <= 0.8091216087341309:
                                                                            return 3
                                                                        else:
                                                                            return 6
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                    return 6
                                                                else:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.17796610854566097:
                                                                        return 4
                                                                    else:
                                                                        return 3
                                                        else:
                                                            return 4
                                                    else:
                                                        if sample.get('ackdat', 0) <= 0.03427726402878761:
                                                            if sample.get('dpkts', 0) <= 0.0016336903208866715:
                                                                if sample.get('ct_srv_src', 0) <= 0.032258063554763794:
                                                                    return 4
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('smean', 0) <= 0.8054054081439972:
                                                                return 6
                                                            else:
                                                                return 3
                                            else:
                                                if sample.get('sload', 0) <= 5.7907545851776376e-05:
                                                    return 3
                                                else:
                                                    if sample.get('dpkts', 0) <= 0.002541296067647636:
                                                        if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                            if sample.get('dtcpb', 0) <= 0.9701928198337555:
                                                                if sample.get('smean', 0) <= 0.7510135173797607:
                                                                    return 3
                                                                else:
                                                                    return 4
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('djit', 0) <= 0.00022919895127415657:
                                                                return 3
                                                            else:
                                                                if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                    return 3
                                                                else:
                                                                    return 3
                                                    else:
                                                        return 3
                                    else:
                                        if sample.get('dur', 0) <= 0.010315268766134977:
                                            if sample.get('stcpb', 0) <= 0.9029608964920044:
                                                if sample.get('dloss', 0) <= 0.00063555475207977:
                                                    if sample.get('dur', 0) <= 0.002603258821181953:
                                                        if sample.get('dur', 0) <= 0.0024333754554390907:
                                                            return 3
                                                        else:
                                                            if sample.get('ackdat', 0) <= 0.016334286890923977:
                                                                return 2
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('dinpkt', 0) <= 0.0004079475475009531:
                                                            return 3
                                                        else:
                                                            if sample.get('ackdat', 0) <= 0.015343771316111088:
                                                                if sample.get('tcprtt', 0) <= 0.02370229735970497:
                                                                    if sample.get('stcpb', 0) <= 0.478618785738945:
                                                                        if sample.get('dur', 0) <= 0.0061875509563833475:
                                                                            if sample.get('ct_flw_http_mthd', 0) <= 0.03125:
                                                                                return 2
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('dur', 0) <= 0.00308514223434031:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                    return 3
                                                                else:
                                                                    return 5
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('dtcpb', 0) <= 0.2801623046398163:
                                                    return 3
                                                else:
                                                    if sample.get('djit', 0) <= 0.00010166196079808287:
                                                        return 3
                                                    else:
                                                        return 2
                                        else:
                                            if sample.get('smean', 0) <= 0.7638513445854187:
                                                if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                                    return 3
                                                else:
                                                    if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                        return 4
                                                    else:
                                                        return 3
                                            else:
                                                if sample.get('dur', 0) <= 0.025846688076853752:
                                                    if sample.get('synack', 0) <= 0.0019787478959187865:
                                                        return 3
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                            else:
                                if sample.get('service', 0) <= 0.3333333283662796:
                                    if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                        if sample.get('dmean', 0) <= 0.043666666373610497:
                                            if sample.get('tcprtt', 0) <= 0.03547697141766548:
                                                if sample.get('sbytes', 0) <= 5.314943518897053e-05:
                                                    if sample.get('dur', 0) <= 0.03198334760963917:
                                                        if sample.get('ackdat', 0) <= 0.01115414034575224:
                                                            if sample.get('dtcpb', 0) <= 0.25840137153863907:
                                                                return 8
                                                            else:
                                                                return 5
                                                        else:
                                                            if sample.get('synack', 0) <= 0.008977658115327358:
                                                                return 3
                                                            else:
                                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                    return 6
                                                                else:
                                                                    if sample.get('sjit', 0) <= 0.004527292447164655:
                                                                        if sample.get('dmean', 0) <= 0.038333332166075706:
                                                                            return 6
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('ackdat', 0) <= 0.019984614104032516:
                                                                            return 4
                                                                        else:
                                                                            if sample.get('stcpb', 0) <= 0.3237980455160141:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                    else:
                                                        if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                                            if sample.get('sinpkt', 0) <= 0.005285415332764387:
                                                                return 8
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('sload', 0) <= 4.5788478075792227e-07:
                                                                return 3
                                                            else:
                                                                return 4
                                                else:
                                                    if sample.get('sbytes', 0) <= 0.00019295403762953356:
                                                        if sample.get('smean', 0) <= 0.13074324280023575:
                                                            if sample.get('dur', 0) <= 0.018218262121081352:
                                                                if sample.get('dbytes', 0) <= 3.875141010212246e-05:
                                                                    if sample.get('tcprtt', 0) <= 0.009200790897011757:
                                                                        return 7
                                                                    else:
                                                                        if sample.get('sbytes', 0) <= 5.454260463011451e-05:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('sinpkt', 0) <= 0.0011638450669124722:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('sbytes', 0) <= 6.338923412840813e-05:
                                                                    if sample.get('dpkts', 0) <= 0.0011798874475061893:
                                                                        if sample.get('ct_srv_src', 0) <= 0.11290322244167328:
                                                                            return 6
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('djit', 0) <= 0.0004622214910341427:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dmean', 0) <= 0.029666665941476822:
                                                                            if sample.get('dtcpb', 0) <= 0.8717166185379028:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('tcprtt', 0) <= 0.03429744951426983:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                        else:
                                                            if sample.get('ct_srv_src', 0) <= 0.08870967477560043:
                                                                if sample.get('smean', 0) <= 0.15810810774564743:
                                                                    return 6
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('synack', 0) <= 0.025439849123358727:
                                                            if sample.get('sjit', 0) <= 0.003342005773447454:
                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                            else:
                                                if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                    if sample.get('sloss', 0) <= 0.000658018427202478:
                                                        if sample.get('sbytes', 0) <= 0.00019692457863129675:
                                                            if sample.get('rate', 0) <= 1.804799831006676e-05:
                                                                if sample.get('ct_dst_ltm', 0) <= 0.043103449046611786:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.04237288050353527:
                                                                        if sample.get('dpkts', 0) <= 0.0011798874475061893:
                                                                            if sample.get('stcpb', 0) <= 0.06353792548179626:
                                                                                if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                                    if sample.get('sloss', 0) <= 0.0004700131685240194:
                                                                                        if sample.get('stcpb', 0) <= 0.057815536856651306:
                                                                                            return 6
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                        return 7
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                if sample.get('dinpkt', 0) <= 0.0035268638748675585:
                                                                                    if sample.get('sload', 0) <= 3.176394216097833e-06:
                                                                                        if sample.get('sbytes', 0) <= 8.345088281203061e-05:
                                                                                            return 6
                                                                                        else:
                                                                                            if sample.get('sbytes', 0) <= 9.585009684087709e-05:
                                                                                                return 3
                                                                                            else:
                                                                                                return 6
                                                                                    else:
                                                                                        if sample.get('tcprtt', 0) <= 0.07287532463669777:
                                                                                            return 6
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    if sample.get('sjit', 0) <= 0.0048738629557192326:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('djit', 0) <= 0.0006925171182956547:
                                                                                            if sample.get('dtcpb', 0) <= 0.7447951436042786:
                                                                                                return 6
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            if sample.get('smean', 0) <= 0.0219594594091177:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('djit', 0) <= 0.0007813622942194343:
                                                                                                    if sample.get('sjit', 0) <= 0.008851957507431507:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    if sample.get('stcpb', 0) <= 0.161185622215271:
                                                                                                        if sample.get('dtcpb', 0) <= 0.2962314933538437:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 6
                                                                                                    else:
                                                                                                        return 6
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.09322033822536469:
                                                                            if sample.get('smean', 0) <= 0.016554053872823715:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('sbytes', 0) <= 4.346690184320323e-05:
                                                                                    if sample.get('sjit', 0) <= 0.004181415191851556:
                                                                                        return 3
                                                                                    else:
                                                                                        return 8
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 6
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('dtcpb', 0) <= 0.40632660686969757:
                                                                    if sample.get('synack', 0) <= 0.026723789051175117:
                                                                        if sample.get('spkts', 0) <= 0.0009394081716891378:
                                                                            return 3
                                                                        else:
                                                                            return 5
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('ackdat', 0) <= 0.011023710016161203:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dur', 0) <= 0.013772327452898026:
                                                                            if sample.get('sload', 0) <= 1.9957789163527195e-06:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                        else:
                                                                            return 6
                                                        else:
                                                            if sample.get('smean', 0) <= 0.20878378301858902:
                                                                return 3
                                                            else:
                                                                if sample.get('dmean', 0) <= 0.03033333271741867:
                                                                    return 6
                                                                else:
                                                                    return 3
                                                    else:
                                                        if sample.get('dbytes', 0) <= 4.3117766836076044e-05:
                                                            if sample.get('synack', 0) <= 0.04340864531695843:
                                                                if sample.get('dbytes', 0) <= 2.1217761968728155e-05:
                                                                    return 3
                                                                else:
                                                                    if sample.get('dtcpb', 0) <= 0.9007967412471771:
                                                                        if sample.get('ackdat', 0) <= 0.032517656683921814:
                                                                            if sample.get('dtcpb', 0) <= 0.029834873043000698:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                        else:
                                                                            if sample.get('djit', 0) <= 0.0005928158934693784:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('dinpkt', 0) <= 0.009479186730459332:
                                                                                    return 6
                                                                                else:
                                                                                    return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                if sample.get('ackdat', 0) <= 0.016524980776011944:
                                                                    return 3
                                                                else:
                                                                    if sample.get('sjit', 0) <= 0.01997203938663006:
                                                                        if sample.get('ackdat', 0) <= 0.06514577008783817:
                                                                            if sample.get('dinpkt', 0) <= 0.0022890917025506496:
                                                                                return 6
                                                                            else:
                                                                                if sample.get('dload', 0) <= 0.00014624903997173533:
                                                                                    if sample.get('dinpkt', 0) <= 0.0030667162500321865:
                                                                                        if sample.get('dur', 0) <= 0.029279696755111217:
                                                                                            if sample.get('sinpkt', 0) <= 0.002146068960428238:
                                                                                                return 3
                                                                                            else:
                                                                                                return 4
                                                                                        else:
                                                                                            return 6
                                                                                    else:
                                                                                        if sample.get('dur', 0) <= 0.03802836500108242:
                                                                                            return 4
                                                                                        else:
                                                                                            if sample.get('dinpkt', 0) <= 0.003220362472347915:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('stcpb', 0) <= 0.07014908362179995:
                                                                                                    return 4
                                                                                                else:
                                                                                                    if sample.get('dtcpb', 0) <= 0.960974395275116:
                                                                                                        if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                                                            if sample.get('stcpb', 0) <= 0.9460141658782959:
                                                                                                                return 6
                                                                                                            else:
                                                                                                                return 3
                                                                                                        else:
                                                                                                            return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                else:
                                                                                    return 4
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                if sample.get('service', 0) <= 0.125:
                                                                    if sample.get('dbytes', 0) <= 5.6762628446449526e-05:
                                                                        if sample.get('dtcpb', 0) <= 0.8390868604183197:
                                                                            return 6
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                        if sample.get('sload', 0) <= 1.93848848084599e-06:
                                                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                if sample.get('dtcpb', 0) <= 0.2666284292936325:
                                                                                    return 6
                                                                                else:
                                                                                    if sample.get('tcprtt', 0) <= 0.08033908531069756:
                                                                                        if sample.get('stcpb', 0) <= 0.1290295198559761:
                                                                                            return 3
                                                                                        else:
                                                                                            return 4
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                return 6
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 4
                                                else:
                                                    if sample.get('sjit', 0) <= 0.0031067864038050175:
                                                        return 3
                                                    else:
                                                        if sample.get('sbytes', 0) <= 0.00017992789798881859:
                                                            if sample.get('smean', 0) <= 0.09324324503540993:
                                                                if sample.get('smean', 0) <= 0.03817567601799965:
                                                                    if sample.get('smean', 0) <= 0.02060810849070549:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('sinpkt', 0) <= 0.0035263439640402794:
                                                                            if sample.get('sbytes', 0) <= 3.879978430632036e-05:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('stcpb', 0) <= 0.022402433678507805:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('rate', 0) <= 1.618974238226656e-05:
                                                                                        return 6
                                                                                    else:
                                                                                        return 3
                                                                        else:
                                                                            if sample.get('tcprtt', 0) <= 0.04918637312948704:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('dmean', 0) <= 0.036333332769572735:
                                                                                    if sample.get('djit', 0) <= 0.0008262000337708741:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('dtcpb', 0) <= 0.4146914631128311:
                                                                                            return 4
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    return 3
                                                                else:
                                                                    if sample.get('dtcpb', 0) <= 0.9390388429164886:
                                                                        if sample.get('ackdat', 0) <= 0.030051782727241516:
                                                                            if sample.get('dload', 0) <= 0.0001272282170248218:
                                                                                if sample.get('djit', 0) <= 0.0004946537083014846:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('tcprtt', 0) <= 0.04987550526857376:
                                                                                        return 6
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('dbytes', 0) <= 2.742617562034866e-05:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 6
                                                            else:
                                                                return 6
                                                        else:
                                                            if sample.get('djit', 0) <= 0.0006262996175792068:
                                                                return 3
                                                            else:
                                                                if sample.get('djit', 0) <= 0.0008560512796975672:
                                                                    return 6
                                                                else:
                                                                    return 3
                                        else:
                                            if sample.get('dbytes', 0) <= 6.262991882977076e-05:
                                                if sample.get('tcprtt', 0) <= 0.04516723193228245:
                                                    if sample.get('dloss', 0) <= 0.00045396767382044345:
                                                        if sample.get('dur', 0) <= 0.016837994568049908:
                                                            if sample.get('synack', 0) <= 0.022669446654617786:
                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('synack', 0) <= 0.030891709960997105:
                                                                if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                    return 6
                                                                else:
                                                                    if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                                                        return 3
                                                                    else:
                                                                        return 4
                                                            else:
                                                                return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('sbytes', 0) <= 6.74294242344331e-05:
                                                        if sample.get('smean', 0) <= 0.032770270481705666:
                                                            return 3
                                                        else:
                                                            return 6
                                                    else:
                                                        if sample.get('dur', 0) <= 0.040621040388941765:
                                                            if sample.get('tcprtt', 0) <= 0.08129238337278366:
                                                                if sample.get('sjit', 0) <= 0.0032625736203044653:
                                                                    return 3
                                                                else:
                                                                    if sample.get('djit', 0) <= 0.00033447748865000904:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('sjit', 0) <= 0.004862130153924227:
                                                                            return 4
                                                                        else:
                                                                            if sample.get('sjit', 0) <= 0.005609260173514485:
                                                                                if sample.get('dbytes', 0) <= 5.116823558637407e-05:
                                                                                    return 3
                                                                                else:
                                                                                    return 6
                                                                            else:
                                                                                if sample.get('synack', 0) <= 0.0465298630297184:
                                                                                    if sample.get('dmean', 0) <= 0.06233333423733711:
                                                                                        if sample.get('ackdat', 0) <= 0.025433986447751522:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('ct_src_ltm', 0) <= 0.0338983042165637:
                                                                                                if sample.get('ackdat', 0) <= 0.05135862156748772:
                                                                                                    return 4
                                                                                                else:
                                                                                                    return 3
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('dbytes', 0) <= 5.6967301134136505e-05:
                                                                if sample.get('sbytes', 0) <= 7.404698408208787e-05:
                                                                    return 3
                                                                else:
                                                                    if sample.get('sload', 0) <= 9.748596880854166e-07:
                                                                        return 6
                                                                    else:
                                                                        return 3
                                                            else:
                                                                return 3
                                            else:
                                                if sample.get('dpkts', 0) <= 0.0011798874475061893:
                                                    if sample.get('dmean', 0) <= 0.10966666787862778:
                                                        if sample.get('ct_src_ltm', 0) <= 0.05932203307747841:
                                                            return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                    else:
                                        if sample.get('dload', 0) <= 9.929978841682896e-05:
                                            if sample.get('sbytes', 0) <= 4.903958324575797e-05:
                                                if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                    if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                        return 3
                                                    else:
                                                        if sample.get('djit', 0) <= 0.0006498990987893194:
                                                            return 3
                                                        else:
                                                            return 3
                                                else:
                                                    return 6
                                            else:
                                                if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                    if sample.get('tcprtt', 0) <= 0.034157320857048035:
                                                        return 3
                                                    else:
                                                        if sample.get('stcpb', 0) <= 0.04195431247353554:
                                                            if sample.get('sinpkt', 0) <= 0.002282517496496439:
                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('ct_dst_ltm', 0) <= 0.06034482829272747:
                                                                if sample.get('dload', 0) <= 8.747998799663037e-05:
                                                                    if sample.get('dur', 0) <= 0.022875046357512474:
                                                                        if sample.get('smean', 0) <= 0.09324324503540993:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 6
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                else:
                                                    if sample.get('synack', 0) <= 0.05097964406013489:
                                                        if sample.get('sjit', 0) <= 0.007060494739562273:
                                                            return 3
                                                        else:
                                                            if sample.get('stcpb', 0) <= 0.6639649569988251:
                                                                if sample.get('sjit', 0) <= 0.008170595858246088:
                                                                    return 3
                                                                else:
                                                                    if sample.get('djit', 0) <= 0.0016717617982067168:
                                                                        return 4
                                                                    else:
                                                                        return 3
                                                            else:
                                                                return 8
                                                    else:
                                                        return 3
                                        else:
                                            if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                if sample.get('rate', 0) <= 1.70201410583104e-05:
                                                    if sample.get('dinpkt', 0) <= 0.0034390180371701717:
                                                        if sample.get('dload', 0) <= 0.00011908895612577908:
                                                            return 3
                                                        else:
                                                            if sample.get('sloss', 0) <= 0.00018800525867845863:
                                                                return 3
                                                            else:
                                                                return 5
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('tcprtt', 0) <= 0.032666921615600586:
                                                        return 3
                                                    else:
                                                        if sample.get('djit', 0) <= 0.0004605342837749049:
                                                            return 6
                                                        else:
                                                            return 3
                                            else:
                                                if sample.get('dmean', 0) <= 0.06533333286643028:
                                                    if sample.get('sloss', 0) <= 0.0004700131685240194:
                                                        if sample.get('dload', 0) <= 0.00016742842854000628:
                                                            if sample.get('djit', 0) <= 0.0005307397805154324:
                                                                if sample.get('ackdat', 0) <= 0.041544971987605095:
                                                                    if sample.get('sinpkt', 0) <= 0.0020556224626488984:
                                                                        if sample.get('dtcpb', 0) <= 0.9509464502334595:
                                                                            return 4
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('rate', 0) <= 2.5939944862329867e-05:
                                                                return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('dur', 0) <= 0.03357659839093685:
                                                            if sample.get('sbytes', 0) <= 0.00027396687073633075:
                                                                if sample.get('dinpkt', 0) <= 0.0012433106894604862:
                                                                    return 3
                                                                else:
                                                                    if sample.get('dtcpb', 0) <= 0.017927279695868492:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dinpkt', 0) <= 0.0037319421535357833:
                                                                            if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                return 4
                                                                            else:
                                                                                if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                                    if sample.get('sloss', 0) <= 0.0008460236713290215:
                                                                                        return 4
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 3
                                                            else:
                                                                if sample.get('sjit', 0) <= 0.0035547083243727684:
                                                                    return 3
                                                                else:
                                                                    return 3
                                                        else:
                                                            if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                if sample.get('djit', 0) <= 0.00046529510291293263:
                                                                    return 3
                                                                else:
                                                                    return 4
                                                            else:
                                                                if sample.get('stcpb', 0) <= 0.8766770660877228:
                                                                    return 6
                                                                else:
                                                                    return 3
                                                else:
                                                    return 3
                                else:
                                    if sample.get('response_body_len', 0) <= 2.288818359375e-05:
                                        if sample.get('smean', 0) <= 0.040878377854824066:
                                            if sample.get('sttl', 0) <= 0.6196078434586525:
                                                if sample.get('djit', 0) <= 0.0004259051929693669:
                                                    if sample.get('dtcpb', 0) <= 0.6699319779872894:
                                                        if sample.get('ackdat', 0) <= 0.016313800122588873:
                                                            if sample.get('dload', 0) <= 0.00043685981654562056:
                                                                return 2
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('synack', 0) <= 0.035220008343458176:
                                                                return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        return 2
                                                else:
                                                    if sample.get('dbytes', 0) <= 7.429627657984383e-05:
                                                        return 2
                                                    else:
                                                        if sample.get('stcpb', 0) <= 0.48528149724006653:
                                                            return 3
                                                        else:
                                                            return 2
                                            else:
                                                if sample.get('sbytes', 0) <= 4.890026684734039e-05:
                                                    if sample.get('tcprtt', 0) <= 0.0400532241910696:
                                                        return 4
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('ct_flw_http_mthd', 0) <= 0.15625:
                                                        if sample.get('synack', 0) <= 0.07055731490254402:
                                                            if sample.get('ct_src_ltm', 0) <= 0.18644067645072937:
                                                                if sample.get('dtcpb', 0) <= 0.024109814781695604:
                                                                    return 3
                                                                else:
                                                                    if sample.get('stcpb', 0) <= 0.565287709236145:
                                                                        return 7
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.6513241231441498:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('dur', 0) <= 0.02072270307689905:
                                                                                return 7
                                                                            else:
                                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                        else:
                                            if sample.get('spkts', 0) <= 0.004790981533005834:
                                                if sample.get('smean', 0) <= 0.049662161618471146:
                                                    if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                                        if sample.get('dbytes', 0) <= 7.477384860976599e-05:
                                                            if sample.get('sloss', 0) <= 0.0007520210783695802:
                                                                if sample.get('dinpkt', 0) <= 0.00273116456810385:
                                                                    if sample.get('dload', 0) <= 0.00010237704918836243:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('ackdat', 0) <= 0.015305871609598398:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('dload', 0) <= 0.0001374489365844056:
                                                                                if sample.get('sjit', 0) <= 0.004848545417189598:
                                                                                    if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                                        if sample.get('smean', 0) <= 0.042229728773236275:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('djit', 0) <= 0.0006692433380521834:
                                                                                                if sample.get('ackdat', 0) <= 0.017557834275066853:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 7
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('rate', 0) <= 2.561945530032972e-05:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                else:
                                                                    if sample.get('sbytes', 0) <= 5.767723632743582e-05:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dtcpb', 0) <= 0.3726344406604767:
                                                                            if sample.get('synack', 0) <= 0.025865660049021244:
                                                                                return 3
                                                                            else:
                                                                                return 7
                                                                        else:
                                                                            return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('djit', 0) <= 0.00031526669044978917:
                                                        if sample.get('smean', 0) <= 0.06182432547211647:
                                                            if sample.get('sinpkt', 0) <= 0.004362053237855434:
                                                                return 7
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('dload', 0) <= 0.00013806772039970383:
                                                                return 3
                                                            else:
                                                                if sample.get('dmean', 0) <= 0.05599999986588955:
                                                                    if sample.get('tcprtt', 0) <= 0.0272313104942441:
                                                                        return 3
                                                                    else:
                                                                        return 2
                                                                else:
                                                                    if sample.get('djit', 0) <= 0.0002863623376470059:
                                                                        return 5
                                                                    else:
                                                                        return 3
                                                    else:
                                                        if sample.get('smean', 0) <= 0.06385134905576706:
                                                            if sample.get('tcprtt', 0) <= 0.1318516619503498:
                                                                if sample.get('stcpb', 0) <= 0.03128940466558561:
                                                                    return 7
                                                                else:
                                                                    if sample.get('ct_srv_src', 0) <= 0.04838709533214569:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('synack', 0) <= 0.04208860360085964:
                                                                if sample.get('ackdat', 0) <= 0.009535888209939003:
                                                                    if sample.get('dload', 0) <= 0.00017922847109730355:
                                                                        return 4
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('ackdat', 0) <= 0.033798908814787865:
                                                                        if sample.get('sbytes', 0) <= 7.739059219602495e-05:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('sloss', 0) <= 0.004324121051467955:
                                                                                if sample.get('dinpkt', 0) <= 0.0019019684405066073:
                                                                                    return 2
                                                                                else:
                                                                                    if sample.get('dmean', 0) <= 0.10399999842047691:
                                                                                        if sample.get('ct_srv_src', 0) <= 0.04838709533214569:
                                                                                            if sample.get('synack', 0) <= 0.02911811973899603:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('dtcpb', 0) <= 0.91057950258255:
                                                                                                    if sample.get('dload', 0) <= 7.650703264516778e-05:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    return 3
                                                                                        else:
                                                                                            if sample.get('ct_flw_http_mthd', 0) <= 0.03125:
                                                                                                return 3
                                                                                            else:
                                                                                                return 2
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                return 3
                                            else:
                                                if sample.get('dur', 0) <= 0.04743261635303497:
                                                    return 4
                                                else:
                                                    return 6
                                    else:
                                        if sample.get('sbytes', 0) <= 5.3288751587388106e-05:
                                            return 3
                                        else:
                                            if sample.get('dload', 0) <= 0.0005085264274384826:
                                                if sample.get('sttl', 0) <= 0.6196078434586525:
                                                    if sample.get('dbytes', 0) <= 9.783366476767696e-05:
                                                        if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                            if sample.get('dbytes', 0) <= 9.196637620334513e-05:
                                                                if sample.get('synack', 0) <= 0.05285007879137993:
                                                                    if sample.get('djit', 0) <= 0.00030587587389163673:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('smean', 0) <= 0.042229728773236275:
                                                                            if sample.get('sinpkt', 0) <= 0.0019986542174592614:
                                                                                if sample.get('sinpkt', 0) <= 0.0019958852790296078:
                                                                                    if sample.get('djit', 0) <= 0.0006593806610908359:
                                                                                        if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                                            return 6
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 6
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('djit', 0) <= 0.0008182245801435784:
                                                                        return 2
                                                                    else:
                                                                        return 3
                                                            else:
                                                                if sample.get('sinpkt', 0) <= 0.0018970359233208:
                                                                    return 3
                                                                else:
                                                                    if sample.get('dtcpb', 0) <= 0.20667393505573273:
                                                                        if sample.get('dur', 0) <= 0.023130645975470543:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('ackdat', 0) <= 0.04534894600510597:
                                                                            return 6
                                                                        else:
                                                                            if sample.get('sload', 0) <= 6.448939586789493e-07:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('response_body_len', 0) <= 3.8242340451688506e-05:
                                                    if sample.get('response_body_len', 0) <= 2.880096417356981e-05:
                                                        return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('tcprtt', 0) <= 0.05718971602618694:
                                                        if sample.get('dinpkt', 0) <= 0.0017687277868390083:
                                                            return 3
                                                        else:
                                                            if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                return 2
                                                            else:
                                                                return 3
                                                    else:
                                                        return 3
                        else:
                            if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                if sample.get('dload', 0) <= 9.056418639374897e-05:
                                    if sample.get('sbytes', 0) <= 7.369869126705453e-05:
                                        if sample.get('sbytes', 0) <= 5.217421494307928e-05:
                                            if sample.get('sbytes', 0) <= 5.092036190035287e-05:
                                                if sample.get('sbytes', 0) <= 4.277031803212594e-05:
                                                    return 3
                                                else:
                                                    return 8
                                            else:
                                                return 4
                                        else:
                                            return 6
                                    else:
                                        if sample.get('dload', 0) <= 5.0967773859156296e-05:
                                            if sample.get('sbytes', 0) <= 0.00012872890874859877:
                                                if sample.get('tcprtt', 0) <= 0.012030464597046375:
                                                    if sample.get('sbytes', 0) <= 0.00010065652895718813:
                                                        return 5
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                            else:
                                                return 3
                                        else:
                                            return 3
                                else:
                                    if sample.get('dur', 0) <= 0.013955152593553066:
                                        if sample.get('sbytes', 0) <= 5.1616947530419566e-05:
                                            return 3
                                        else:
                                            if sample.get('sload', 0) <= 1.6473655932713882e-06:
                                                return 3
                                            else:
                                                if sample.get('djit', 0) <= 0.00011297736273263581:
                                                    if sample.get('dload', 0) <= 0.0009894086106214672:
                                                        return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('ct_dst_ltm', 0) <= 0.0775862094014883:
                                                        if sample.get('dtcpb', 0) <= 0.9734700322151184:
                                                            if sample.get('ackdat', 0) <= 0.041088467463850975:
                                                                if sample.get('ackdat', 0) <= 0.02179031725972891:
                                                                    if sample.get('dtcpb', 0) <= 0.9324856996536255:
                                                                        if sample.get('djit', 0) <= 0.0001172765769297257:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 9
                                                                else:
                                                                    if sample.get('dload', 0) <= 0.00023627080372534692:
                                                                        if sample.get('synack', 0) <= 0.03051021695137024:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('dinpkt', 0) <= 0.0018412885256111622:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                            if sample.get('dmean', 0) <= 0.04766666702926159:
                                                if sample.get('dload', 0) <= 0.0001299197756452486:
                                                    if sample.get('sbytes', 0) <= 5.969733501842711e-05:
                                                        if sample.get('dtcpb', 0) <= 0.30505987256765366:
                                                            return 3
                                                        else:
                                                            return 6
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('stcpb', 0) <= 0.9608882367610931:
                                                        return 6
                                                    else:
                                                        return 3
                                            else:
                                                if sample.get('ackdat', 0) <= 0.029928864911198616:
                                                    if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                        return 3
                                                    else:
                                                        return 3
                                                else:
                                                    return 4
                                        else:
                                            if sample.get('smean', 0) <= 0.032770270481705666:
                                                return 3
                                            else:
                                                if sample.get('sttl', 0) <= 0.6196078434586525:
                                                    return 3
                                                else:
                                                    if sample.get('djit', 0) <= 0.00029639265267178416:
                                                        if sample.get('rate', 0) <= 4.954959877068177e-05:
                                                            return 3
                                                        else:
                                                            return 6
                                                    else:
                                                        if sample.get('dload', 0) <= 0.0001237535907421261:
                                                            return 3
                                                        else:
                                                            return 4
                            else:
                                if sample.get('sttl', 0) <= 0.6196078434586525:
                                    if sample.get('sbytes', 0) <= 5.9000747569371015e-05:
                                        return 0
                                    else:
                                        if sample.get('sinpkt', 0) <= 0.0045364456018432975:
                                            return 3
                                        else:
                                            return 6
                                else:
                                    if sample.get('dmean', 0) <= 0.07666666433215141:
                                        if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                            if sample.get('tcprtt', 0) <= 0.08605246990919113:
                                                if sample.get('sbytes', 0) <= 0.014054612256586552:
                                                    if sample.get('dloss', 0) <= 0.00127110950415954:
                                                        if sample.get('sbytes', 0) <= 4.0750221160124056e-05:
                                                            if sample.get('dmean', 0) <= 0.03933333232998848:
                                                                return 4
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('ct_flw_http_mthd', 0) <= 0.03125:
                                                                if sample.get('ct_srv_src', 0) <= 0.2177419289946556:
                                                                    if sample.get('dbytes', 0) <= 4.468692532100249e-05:
                                                                        if sample.get('dmean', 0) <= 0.03766666539013386:
                                                                            if sample.get('dload', 0) <= 0.0002953137445729226:
                                                                                if sample.get('synack', 0) <= 0.026900744065642357:
                                                                                    if sample.get('rate', 0) <= 0.4166666567325592:
                                                                                        if sample.get('sload', 0) <= 0.055199265480041504:
                                                                                            return 4
                                                                                        else:
                                                                                            if sample.get('sload', 0) <= 0.05934232473373413:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                                                    return 4
                                                                                                else:
                                                                                                    if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                                                                                        if sample.get('sbytes', 0) <= 8.421712482231669e-05:
                                                                                                            return 4
                                                                                                        else:
                                                                                                            if sample.get('sinpkt', 0) <= 1.249791825586044e-07:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                return 3
                                                                                                    else:
                                                                                                        return 4
                                                                                    else:
                                                                                        if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                                            return 4
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    if sample.get('synack', 0) <= 0.026935763657093048:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('synack', 0) <= 0.027235752902925014:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('sload', 0) <= 1.219561681864434e-05:
                                                                                                if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                                                    if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                                        if sample.get('dload', 0) <= 0.00012046460687997751:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 4
                                                                                                    else:
                                                                                                        return 4
                                                                                                else:
                                                                                                    if sample.get('sload', 0) <= 3.1121458050620276e-06:
                                                                                                        return 9
                                                                                                    else:
                                                                                                        return 4
                                                                                            else:
                                                                                                if sample.get('tcprtt', 0) <= 0.04268820956349373:
                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('sjit', 0) <= 0.0023238519206643105:
                                                                                                        return 4
                                                                                                    else:
                                                                                                        if sample.get('dpkts', 0) <= 0.0009076057467609644:
                                                                                                            return 5
                                                                                                        else:
                                                                                                            return 3
                                                                            else:
                                                                                if sample.get('dload', 0) <= 0.00031508180836681277:
                                                                                    if sample.get('sinpkt', 0) <= 0.0005885919963475317:
                                                                                        return 3
                                                                                    else:
                                                                                        return 4
                                                                                else:
                                                                                    return 4
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('dtcpb', 0) <= 0.988648921251297:
                                                                            return 4
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('dbytes', 0) <= 3.295234364486532e-05:
                                                                    if sample.get('dinpkt', 0) <= 0.0014947245654184371:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 4
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                            else:
                                                return 3
                                        else:
                                            if sample.get('sload', 0) <= 0.1829916387796402:
                                                if sample.get('smean', 0) <= 0.04560810886323452:
                                                    if sample.get('ct_src_ltm', 0) <= 0.05932203307747841:
                                                        if sample.get('dinpkt', 0) <= 0.001220277335960418:
                                                            return 3
                                                        else:
                                                            if sample.get('tcprtt', 0) <= 0.057268351316452026:
                                                                return 4
                                                            else:
                                                                if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                                                    return 3
                                                                else:
                                                                    return 4
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('sload', 0) <= 3.951417284042691e-06:
                                                        if sample.get('sloss', 0) <= 0.0004700131685240194:
                                                            if sample.get('dtcpb', 0) <= 0.6279013156890869:
                                                                return 3
                                                            else:
                                                                if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                    return 4
                                                                else:
                                                                    return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('sbytes', 0) <= 5.983665141684469e-05:
                                                            return 3
                                                        else:
                                                            if sample.get('stcpb', 0) <= 0.8048198223114014:
                                                                if sample.get('sload', 0) <= 0.07854550704360008:
                                                                    return 4
                                                                else:
                                                                    if sample.get('rate', 0) <= 0.138888880610466:
                                                                        return 3
                                                                    else:
                                                                        return 4
                                                            else:
                                                                if sample.get('stcpb', 0) <= 0.8619284927845001:
                                                                    return 3
                                                                else:
                                                                    return 4
                                            else:
                                                if sample.get('sbytes', 0) <= 0.0001567316212458536:
                                                    return 5
                                                else:
                                                    return 3
                                    else:
                                        return 3
                else:
                    if sample.get('sloss', 0) <= 9.400262933922932e-05:
                        if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                            if sample.get('proto', 0) <= 0.8846153616905212:
                                if sample.get('sbytes', 0) <= 5.085070597488084e-06:
                                    return 3
                                else:
                                    if sample.get('sjit', 0) <= 1.760773557180073e-05:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                            if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                return 2
                                            else:
                                                return 3
                                        else:
                                            if sample.get('rate', 0) <= 0.16250000149011612:
                                                if sample.get('proto', 0) <= 0.29615385830402374:
                                                    if sample.get('proto', 0) <= 0.10000000149011612:
                                                        if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                            return 3
                                                        else:
                                                            return 2
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('proto', 0) <= 0.8192307651042938:
                                                        if sample.get('proto', 0) <= 0.5730769336223602:
                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                                                return 2
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('proto', 0) <= 0.6500000059604645:
                                                                if sample.get('sload', 0) <= 0.017927950248122215:
                                                                    return 3
                                                                else:
                                                                    return 2
                                                            else:
                                                                return 2
                                                    else:
                                                        return 3
                                            else:
                                                if sample.get('ct_src_ltm', 0) <= 0.05084745865315199:
                                                    if sample.get('proto', 0) <= 0.29615385830402374:
                                                        return 2
                                                    else:
                                                        if sample.get('proto', 0) <= 0.7192307710647583:
                                                            if sample.get('proto', 0) <= 0.45384615659713745:
                                                                return 2
                                                            else:
                                                                return 3
                                                        else:
                                                            return 2
                                                else:
                                                    return 2
                                    else:
                                        if sample.get('ct_srv_dst', 0) <= 0.032786885276436806:
                                            return 3
                                        else:
                                            return 4
                            else:
                                if sample.get('proto', 0) <= 0.9038461446762085:
                                    if sample.get('service', 0) <= 0.0416666679084301:
                                        if sample.get('rate', 0) <= 0.08893280476331711:
                                            if sample.get('rate', 0) <= 0.07417582347989082:
                                                if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                                    if sample.get('sload', 0) <= 0.0023916607606224716:
                                                        if sample.get('dload', 0) <= 0.0003056437417399138:
                                                            if sample.get('dur', 0) <= 0.0002517917091608979:
                                                                if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                    return 3
                                                                else:
                                                                    return 2
                                                            else:
                                                                if sample.get('spkts', 0) <= 0.003992484678747132:
                                                                    return 2
                                                                else:
                                                                    return 3
                                                        else:
                                                            if sample.get('dur', 0) <= 0.00020427504205144942:
                                                                return 2
                                                            else:
                                                                return 3
                                                    else:
                                                        return 6
                                                else:
                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                                        if sample.get('rate', 0) <= 0.02817460149526596:
                                                            return 4
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('ct_srv_dst', 0) <= 0.6065573692321777:
                                                            if sample.get('smean', 0) <= 0.6689189374446869:
                                                                if sample.get('ct_srv_src', 0) <= 0.18548386543989182:
                                                                    if sample.get('sinpkt', 0) <= 1.1664724297588691e-07:
                                                                        return 3
                                                                    else:
                                                                        return 6
                                                                else:
                                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.3467741906642914:
                                                                        if sample.get('sload', 0) <= 0.01508960803039372:
                                                                            if sample.get('ct_srv_src', 0) <= 0.3306451588869095:
                                                                                if sample.get('ct_srv_src', 0) <= 0.3145161271095276:
                                                                                    return 3
                                                                                else:
                                                                                    return 6
                                                                            else:
                                                                                return 4
                                                                        else:
                                                                            return 6
                                                                    else:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.2118644043803215:
                                                                            return 6
                                                                        else:
                                                                            return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                            else:
                                                if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                    return 6
                                                else:
                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.35483870655298233:
                                                        if sample.get('sbytes', 0) <= 1.1702627716658753e-05:
                                                            return 6
                                                        else:
                                                            if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                return 3
                                                            else:
                                                                if sample.get('ct_srv_dst', 0) <= 0.23770491778850555:
                                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.06451612710952759:
                                                                        if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        if sample.get('ct_dst_ltm', 0) <= 0.12931034713983536:
                                                                            return 6
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    return 4
                                                    else:
                                                        return 6
                                        else:
                                            if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                                                if sample.get('sbytes', 0) <= 7.80175173531461e-06:
                                                    return 6
                                                else:
                                                    if sample.get('smean', 0) <= 0.049662161618471146:
                                                        return 7
                                                    else:
                                                        if sample.get('sload', 0) <= 0.0988990031182766:
                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.08870967477560043:
                                                                return 2
                                                            else:
                                                                return 3
                                                        else:
                                                            return 6
                                            else:
                                                if sample.get('ct_srv_dst', 0) <= 0.5655737519264221:
                                                    if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                        if sample.get('smean', 0) <= 0.023648648522794247:
                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.05645161122083664:
                                                                if sample.get('sload', 0) <= 0.01776765286922455:
                                                                    if sample.get('sload', 0) <= 0.0077448743395507336:
                                                                        if sample.get('sload', 0) <= 0.007498101564124227:
                                                                            return 6
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                            return 6
                                                                        else:
                                                                            if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                                                if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                    return 6
                                                                                else:
                                                                                    if sample.get('sload', 0) <= 0.00932253384962678:
                                                                                        return 6
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                return 6
                                                                else:
                                                                    if sample.get('rate', 0) <= 0.2916666567325592:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                            return 6
                                                                        else:
                                                                            if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                                                if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                                                                    return 6
                                                                                else:
                                                                                    if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                                                                        return 4
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                return 6
                                                            else:
                                                                if sample.get('sbytes', 0) <= 4.667119469559111e-06:
                                                                    return 6
                                                                else:
                                                                    if sample.get('ct_srv_dst', 0) <= 0.07377048954367638:
                                                                        if sample.get('ct_srv_src', 0) <= 0.11290322244167328:
                                                                            if sample.get('ct_src_ltm', 0) <= 0.05932203494012356:
                                                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                                    if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                                                                        return 3
                                                                                    else:
                                                                                        return 4
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 6
                                                                    else:
                                                                        return 6
                                                        else:
                                                            if sample.get('sbytes', 0) <= 1.0100482086272677e-05:
                                                                if sample.get('sbytes', 0) <= 9.89150657915161e-06:
                                                                    return 3
                                                                else:
                                                                    return 7
                                                            else:
                                                                if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                                                        if sample.get('sbytes', 0) <= 6.735976785421371e-05:
                                                                            if sample.get('sbytes', 0) <= 6.652386582572944e-05:
                                                                                if sample.get('smean', 0) <= 0.2408783808350563:
                                                                                    return 6
                                                                                else:
                                                                                    if sample.get('sload', 0) <= 0.05855594202876091:
                                                                                        return 3
                                                                                    else:
                                                                                        return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 6
                                                                    else:
                                                                        if sample.get('smean', 0) <= 0.348986491560936:
                                                                            return 6
                                                                        else:
                                                                            if sample.get('smean', 0) <= 0.3587837964296341:
                                                                                if sample.get('ct_srv_dst', 0) <= 0.09836065396666527:
                                                                                    if sample.get('sload', 0) <= 0.1111617237329483:
                                                                                        if sample.get('sload', 0) <= 0.07958169281482697:
                                                                                            return 3
                                                                                        else:
                                                                                            return 6
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 6
                                                                            else:
                                                                                return 6
                                                                else:
                                                                    if sample.get('sbytes', 0) <= 2.7097155907540582e-05:
                                                                        if sample.get('sload', 0) <= 0.039028093218803406:
                                                                            if sample.get('rate', 0) <= 0.10555555298924446:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('rate', 0) <= 0.15555555373430252:
                                                                                    return 6
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 8
                                                                    else:
                                                                        if sample.get('sbytes', 0) <= 5.064172910351772e-05:
                                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                        else:
                                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                                                                if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                    if sample.get('smean', 0) <= 0.38445945084095:
                                                                                        return 6
                                                                                    else:
                                                                                        if sample.get('sload', 0) <= 0.12424932047724724:
                                                                                            return 3
                                                                                        else:
                                                                                            return 6
                                                                                else:
                                                                                    return 4
                                                                            else:
                                                                                return 6
                                                    else:
                                                        if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                                            if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                                                if sample.get('sbytes', 0) <= 0.00019998955394839868:
                                                                    if sample.get('rate', 0) <= 0.09545454755425453:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.1440677996724844:
                                                                            return 4
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('sload', 0) <= 0.2586686685681343:
                                                                            if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('ct_srv_src', 0) <= 0.032258063554763794:
                                                                    return 3
                                                                else:
                                                                    if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                                                                        if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                return 4
                                                                            else:
                                                                                if sample.get('sload', 0) <= 0.023842064663767815:
                                                                                    return 4
                                                                                else:
                                                                                    return 7
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        return 3
                                                        else:
                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.2338709607720375:
                                                                if sample.get('ct_srv_src', 0) <= 0.2177419289946556:
                                                                    if sample.get('ct_src_dport_ltm', 0) <= 0.18103448301553726:
                                                                        if sample.get('ct_dst_src_ltm', 0) <= 0.05645161122083664:
                                                                            if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                                                if sample.get('sbytes', 0) <= 0.00018410741176921874:
                                                                                    if sample.get('rate', 0) <= 0.4166666567325592:
                                                                                        if sample.get('smean', 0) <= 0.341891884803772:
                                                                                            return 6
                                                                                        else:
                                                                                            if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                                                                if sample.get('sload', 0) <= 0.16628700494766235:
                                                                                                    return 6
                                                                                                else:
                                                                                                    if sample.get('sload', 0) <= 0.1864844188094139:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 6
                                                                                            else:
                                                                                                if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 6
                                                                                    else:
                                                                                        if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                                                            if sample.get('ct_src_ltm', 0) <= 0.0338983042165637:
                                                                                                return 3
                                                                                            else:
                                                                                                return 4
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                                                    return 6
                                                                                else:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('ct_dst_ltm', 0) <= 0.025862068869173527:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('sbytes', 0) <= 4.667119469559111e-06:
                                                                                                if sample.get('ct_srv_dst', 0) <= 0.08196721225976944:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 4
                                                                                            else:
                                                                                                return 4
                                                                        else:
                                                                            if sample.get('ct_srv_dst', 0) <= 0.2049180343747139:
                                                                                if sample.get('sload', 0) <= 0.41040240228176117:
                                                                                    if sample.get('sload', 0) <= 0.006396538810804486:
                                                                                        if sample.get('sload', 0) <= 0.006067508365958929:
                                                                                            return 6
                                                                                        else:
                                                                                            return 4
                                                                                    else:
                                                                                        if sample.get('sload', 0) <= 0.03933027759194374:
                                                                                            if sample.get('ct_src_dport_ltm', 0) <= 0.06034482829272747:
                                                                                                return 6
                                                                                            else:
                                                                                                if sample.get('sbytes', 0) <= 4.667119469559111e-06:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 6
                                                                                        else:
                                                                                            if sample.get('sload', 0) <= 0.03941464610397816:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('sload', 0) <= 0.04409010149538517:
                                                                                                    if sample.get('sload', 0) <= 0.042672738432884216:
                                                                                                        if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 6
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    if sample.get('ct_dst_ltm', 0) <= 0.025862068869173527:
                                                                                                        if sample.get('ct_srv_src', 0) <= 0.08870967477560043:
                                                                                                            if sample.get('ct_src_ltm', 0) <= 0.04237288050353527:
                                                                                                                return 6
                                                                                                            else:
                                                                                                                return 3
                                                                                                        else:
                                                                                                            if sample.get('ct_srv_src', 0) <= 0.12096773833036423:
                                                                                                                if sample.get('sload', 0) <= 0.09202732890844345:
                                                                                                                    return 3
                                                                                                                else:
                                                                                                                    if sample.get('sbytes', 0) <= 9.877575212158263e-05:
                                                                                                                        return 4
                                                                                                                    else:
                                                                                                                        return 3
                                                                                                            else:
                                                                                                                return 6
                                                                                                    else:
                                                                                                        if sample.get('ct_srv_dst', 0) <= 0.07377048954367638:
                                                                                                            if sample.get('sload', 0) <= 0.19102757424116135:
                                                                                                                return 6
                                                                                                            else:
                                                                                                                return 3
                                                                                                        else:
                                                                                                            return 6
                                                                                else:
                                                                                    if sample.get('sload', 0) <= 0.4859528988599777:
                                                                                        return 4
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                return 4
                                                                    else:
                                                                        return 4
                                                                else:
                                                                    if sample.get('ct_dst_ltm', 0) <= 0.3017241284251213:
                                                                        if sample.get('ct_dst_ltm', 0) <= 0.025862068869173527:
                                                                            return 6
                                                                        else:
                                                                            if sample.get('ct_srv_dst', 0) <= 0.23770491778850555:
                                                                                if sample.get('ct_dst_src_ltm', 0) <= 0.10483870655298233:
                                                                                    if sample.get('ct_srv_dst', 0) <= 0.0901639349758625:
                                                                                        if sample.get('sload', 0) <= 0.018507972126826644:
                                                                                            return 4
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 4
                                                                                else:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.16101694852113724:
                                                                                        return 6
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                return 4
                                                                    else:
                                                                        return 3
                                                            else:
                                                                if sample.get('ct_dst_src_ltm', 0) <= 0.44354838132858276:
                                                                    if sample.get('ct_srv_dst', 0) <= 0.4262295067310333:
                                                                        if sample.get('ct_srv_src', 0) <= 0.3145161271095276:
                                                                            if sample.get('ct_srv_src', 0) <= 0.2983870953321457:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 6
                                                                    else:
                                                                        if sample.get('ct_srv_src', 0) <= 0.44354838132858276:
                                                                            return 4
                                                                        else:
                                                                            if sample.get('ct_src_ltm', 0) <= 0.22033898532390594:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                else:
                                                                    return 6
                                                else:
                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.6209677457809448:
                                                        if sample.get('ct_srv_dst', 0) <= 0.6065573692321777:
                                                            if sample.get('ct_srv_src', 0) <= 0.6048387289047241:
                                                                return 4
                                                            else:
                                                                return 6
                                                        else:
                                                            return 4
                                                    else:
                                                        return 6
                                    else:
                                        if sample.get('sbytes', 0) <= 6.338923412840813e-06:
                                            return 5
                                        else:
                                            if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                                if sample.get('sbytes', 0) <= 8.567995791963767e-06:
                                                    return 3
                                                else:
                                                    return 5
                                            else:
                                                return 3
                                else:
                                    if sample.get('sload', 0) <= 0.10124018602073193:
                                        if sample.get('proto', 0) <= 0.949999988079071:
                                            if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                if sample.get('rate', 0) <= 0.16250000149011612:
                                                    if sample.get('rate', 0) <= 0.1180555522441864:
                                                        if sample.get('ct_dst_src_ltm', 0) <= 0.04838709533214569:
                                                            return 3
                                                        else:
                                                            return 2
                                                    else:
                                                        return 2
                                                else:
                                                    return 2
                                            else:
                                                if sample.get('rate', 0) <= 0.1180555522441864:
                                                    return 3
                                                else:
                                                    if sample.get('rate', 0) <= 0.1339285671710968:
                                                        if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                            if sample.get('ct_src_ltm', 0) <= 0.22033897787332535:
                                                                if sample.get('proto', 0) <= 0.9230769276618958:
                                                                    return 3
                                                                else:
                                                                    return 2
                                                            else:
                                                                return 2
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('sload', 0) <= 0.03796506766229868:
                                                            return 2
                                                        else:
                                                            return 3
                                        else:
                                            return 2
                                    else:
                                        return 2
                        else:
                            if sample.get('service', 0) <= 0.1250000037252903:
                                if sample.get('sbytes', 0) <= 6.965850161577691e-06:
                                    if sample.get('proto', 0) <= 0.8269230723381042:
                                        if sample.get('proto', 0) <= 0.35384616255760193:
                                            return 3
                                        else:
                                            if sample.get('sbytes', 0) <= 4.94575351694948e-06:
                                                if sample.get('smean', 0) <= 0.011486486066132784:
                                                    return 3
                                                else:
                                                    return 2
                                            else:
                                                return 2
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.2016128972172737:
                                            if sample.get('rate', 0) <= 0.10555555298924446:
                                                if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                    return 3
                                                else:
                                                    return 4
                                            else:
                                                if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                                    if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                        if sample.get('smean', 0) <= 0.014527026563882828:
                                                            return 3
                                                        else:
                                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                if sample.get('sload', 0) <= 0.012221136596053839:
                                                                    if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                        return 4
                                                                    else:
                                                                        return 6
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 6
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                                        return 3
                                                    else:
                                                        if sample.get('ct_dst_ltm', 0) <= 0.043103449046611786:
                                                            return 6
                                                        else:
                                                            if sample.get('ct_srv_dst', 0) <= 0.08196721225976944:
                                                                return 3
                                                            else:
                                                                return 6
                                        else:
                                            if sample.get('ct_dst_src_ltm', 0) <= 0.10483870655298233:
                                                return 4
                                            else:
                                                return 6
                                else:
                                    if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                        if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                            if sample.get('sbytes', 0) <= 1.1493652436911361e-05:
                                                if sample.get('smean', 0) <= 0.037837836891412735:
                                                    if sample.get('dur', 0) <= 0.0007882001509500469:
                                                        if sample.get('ct_srv_src', 0) <= 0.016129031777381897:
                                                            return 2
                                                        else:
                                                            return 1
                                                    else:
                                                        return 6
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('smean', 0) <= 0.31824323534965515:
                                                    if sample.get('sbytes', 0) <= 8.045556751312688e-05:
                                                        if sample.get('sload', 0) <= 0.01602969691157341:
                                                            if sample.get('dload', 0) <= 7.594419912493322e-05:
                                                                if sample.get('sload', 0) <= 2.515570258765365e-07:
                                                                    return 3
                                                                else:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.0762711875140667:
                                                                        return 3
                                                                    else:
                                                                        return 4
                                                            else:
                                                                if sample.get('dbytes', 0) <= 3.192897929693572e-05:
                                                                    return 3
                                                                else:
                                                                    return 2
                                                        else:
                                                            if sample.get('ct_src_ltm', 0) <= 0.12711864709854126:
                                                                if sample.get('rate', 0) <= 0.2666666582226753:
                                                                    if sample.get('smean', 0) <= 0.11081080697476864:
                                                                        if sample.get('proto', 0) <= 0.9538461565971375:
                                                                            if sample.get('proto', 0) <= 0.6923076957464218:
                                                                                return 2
                                                                            else:
                                                                                if sample.get('sload', 0) <= 0.024677296169102192:
                                                                                    return 3
                                                                                else:
                                                                                    return 2
                                                                        else:
                                                                            return 2
                                                                    else:
                                                                        return 2
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 2
                                                    else:
                                                        if sample.get('sbytes', 0) <= 0.00020319384202593938:
                                                            if sample.get('sjit', 0) <= 0.0003303360936115496:
                                                                return 2
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('sinpkt', 0) <= 0.0021453843219205737:
                                                                return 3
                                                            else:
                                                                if sample.get('dur', 0) <= 0.4776255935430527:
                                                                    return 3
                                                                else:
                                                                    return 1
                                                else:
                                                    return 3
                                        else:
                                            if sample.get('state', 0) <= 0.4166666716337204:
                                                return 3
                                            else:
                                                if sample.get('sbytes', 0) <= 1.0030823887063889e-05:
                                                    return 2
                                                else:
                                                    if sample.get('smean', 0) <= 0.27162162214517593:
                                                        if sample.get('smean', 0) <= 0.01756756752729416:
                                                            return 3
                                                        else:
                                                            if sample.get('sbytes', 0) <= 0.0006595266750082374:
                                                                if sample.get('sinpkt', 0) <= 0.013088207226246595:
                                                                    if sample.get('sbytes', 0) <= 3.0301447623060085e-05:
                                                                        if sample.get('proto', 0) <= 0.9538461565971375:
                                                                            if sample.get('ct_srv_dst', 0) <= 0.13934426009655:
                                                                                if sample.get('smean', 0) <= 0.03277026955038309:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('sload', 0) <= 0.04429258033633232:
                                                                                        if sample.get('sload', 0) <= 0.026322447694838047:
                                                                                            if sample.get('rate', 0) <= 0.11250000074505806:
                                                                                                if sample.get('ct_srv_src', 0) <= 0.09677419066429138:
                                                                                                    if sample.get('sload', 0) <= 0.014495754148811102:
                                                                                                        return 0
                                                                                                    else:
                                                                                                        if sample.get('ct_src_ltm', 0) <= 0.05084745865315199:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 0
                                                                                                else:
                                                                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                                                                                        if sample.get('proto', 0) <= 0.5576923191547394:
                                                                                                            if sample.get('proto', 0) <= 0.10384615557268262:
                                                                                                                return 4
                                                                                                            else:
                                                                                                                return 3
                                                                                                        else:
                                                                                                            return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                            else:
                                                                                                if sample.get('sload', 0) <= 0.023502185009419918:
                                                                                                    if sample.get('proto', 0) <= 0.47307692654430866:
                                                                                                        return 4
                                                                                                    else:
                                                                                                        if sample.get('ct_dst_ltm', 0) <= 0.025862068869173527:
                                                                                                            if sample.get('ct_src_ltm', 0) <= 0.06779661029577255:
                                                                                                                return 4
                                                                                                            else:
                                                                                                                return 0
                                                                                                        else:
                                                                                                            if sample.get('ct_srv_dst', 0) <= 0.10655737295746803:
                                                                                                                if sample.get('ct_srv_src', 0) <= 0.06451612710952759:
                                                                                                                    return 0
                                                                                                                else:
                                                                                                                    return 4
                                                                                                            else:
                                                                                                                return 3
                                                                                                else:
                                                                                                    return 3
                                                                                        else:
                                                                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                                return 4
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        if sample.get('rate', 0) <= 0.4166666567325592:
                                                                                            return 4
                                                                                        else:
                                                                                            if sample.get('ct_srv_dst', 0) <= 0.1147540956735611:
                                                                                                return 3
                                                                                            else:
                                                                                                return 0
                                                                            else:
                                                                                if sample.get('ct_srv_src', 0) <= 0.18548386543989182:
                                                                                    if sample.get('ct_srv_src', 0) <= 0.16935483366250992:
                                                                                        if sample.get('sload', 0) <= 0.02784105110913515:
                                                                                            if sample.get('rate', 0) <= 0.1547618955373764:
                                                                                                if sample.get('ct_src_ltm', 0) <= 0.050847457721829414:
                                                                                                    if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                                        if sample.get('proto', 0) <= 0.9230769276618958:
                                                                                                            if sample.get('ct_srv_dst', 0) <= 0.15573770552873611:
                                                                                                                if sample.get('proto', 0) <= 0.7423076927661896:
                                                                                                                    return 4
                                                                                                                else:
                                                                                                                    return 3
                                                                                                            else:
                                                                                                                return 3
                                                                                                        else:
                                                                                                            return 1
                                                                                                    else:
                                                                                                        if sample.get('ct_srv_src', 0) <= 0.15322580188512802:
                                                                                                            return 1
                                                                                                        else:
                                                                                                            return 3
                                                                                                else:
                                                                                                    if sample.get('rate', 0) <= 0.09545454755425453:
                                                                                                        if sample.get('ct_srv_src', 0) <= 0.15322580188512802:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 4
                                                                                                    else:
                                                                                                        if sample.get('ct_dst_ltm', 0) <= 0.025862068869173527:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 2
                                                                                            else:
                                                                                                return 0
                                                                                        else:
                                                                                            if sample.get('ct_src_ltm', 0) <= 0.050847457721829414:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('ct_srv_dst', 0) <= 0.15573770552873611:
                                                                                                    return 4
                                                                                                else:
                                                                                                    return 3
                                                                                    else:
                                                                                        if sample.get('rate', 0) <= 0.12142856791615486:
                                                                                            if sample.get('proto', 0) <= 0.09615384787321091:
                                                                                                return 4
                                                                                            else:
                                                                                                if sample.get('proto', 0) <= 0.38846154883503914:
                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('ct_src_ltm', 0) <= 0.12711864337325096:
                                                                                                        if sample.get('proto', 0) <= 0.8692307770252228:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            if sample.get('proto', 0) <= 0.9230769276618958:
                                                                                                                return 4
                                                                                                            else:
                                                                                                                return 3
                                                                                                    else:
                                                                                                        return 4
                                                                                        else:
                                                                                            if sample.get('proto', 0) <= 0.35769231617450714:
                                                                                                return 3
                                                                                            else:
                                                                                                return 4
                                                                                else:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.09322033822536469:
                                                                                        if sample.get('ct_src_ltm', 0) <= 0.04237288050353527:
                                                                                            if sample.get('sload', 0) <= 0.01533942250534892:
                                                                                                return 4
                                                                                            else:
                                                                                                if sample.get('rate', 0) <= 0.15555555373430252:
                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('sload', 0) <= 0.03416856378316879:
                                                                                                        return 4
                                                                                                    else:
                                                                                                        if sample.get('proto', 0) <= 0.7230769395828247:
                                                                                                            return 1
                                                                                                        else:
                                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('sload', 0) <= 0.026575549505650997:
                                                                                                return 0
                                                                                            else:
                                                                                                if sample.get('ct_src_ltm', 0) <= 0.06779661029577255:
                                                                                                    return 1
                                                                                                else:
                                                                                                    return 4
                                                                                    else:
                                                                                        return 3
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('sjit', 0) <= 0.0010439167963340878:
                                                                        return 4
                                                                    else:
                                                                        return 2
                                                            else:
                                                                if sample.get('dur', 0) <= 0.9869866371154785:
                                                                    return 1
                                                                else:
                                                                    return 3
                                                    else:
                                                        if sample.get('ct_srv_dst', 0) <= 0.08196721225976944:
                                                            return 6
                                                        else:
                                                            return 3
                                    else:
                                        if sample.get('smean', 0) <= 0.0763513520359993:
                                            if sample.get('smean', 0) <= 0.033445945009589195:
                                                return 6
                                            else:
                                                if sample.get('ct_src_ltm', 0) <= 0.37288135290145874:
                                                    if sample.get('proto', 0) <= 0.9769230782985687:
                                                        if sample.get('ct_dst_src_ltm', 0) <= 0.19354838877916336:
                                                            if sample.get('proto', 0) <= 0.5807692408561707:
                                                                return 2
                                                            else:
                                                                if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                                                                    if sample.get('sbytes', 0) <= 0.000193371987734281:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.24576270580291748:
                                                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                if sample.get('rate', 0) <= 0.1180555522441864:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('rate', 0) <= 0.2666666582226753:
                                                                                        if sample.get('sload', 0) <= 0.024677296169102192:
                                                                                            if sample.get('proto', 0) <= 0.6730769276618958:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('proto', 0) <= 0.8192307651042938:
                                                                                                    return 2
                                                                                                else:
                                                                                                    return 3
                                                                                        else:
                                                                                            return 2
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                if sample.get('ct_src_ltm', 0) <= 0.12711864337325096:
                                                                                    if sample.get('proto', 0) <= 0.7346153855323792:
                                                                                        if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                                                            return 2
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('ct_src_ltm', 0) <= 0.05932203307747841:
                                                                                            if sample.get('sload', 0) <= 0.024677296169102192:
                                                                                                return 3
                                                                                            else:
                                                                                                return 2
                                                                                        else:
                                                                                            return 2
                                                                                else:
                                                                                    if sample.get('sload', 0) <= 0.02784105110913515:
                                                                                        if sample.get('rate', 0) <= 0.1458333283662796:
                                                                                            return 2
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 2
                                                                        else:
                                                                            return 2
                                                                    else:
                                                                        return 2
                                                                else:
                                                                    if sample.get('sload', 0) <= 0.017927950248122215:
                                                                        return 2
                                                                    else:
                                                                        if sample.get('ct_srv_dst', 0) <= 0.0901639349758625:
                                                                            if sample.get('sload', 0) <= 0.10124018602073193:
                                                                                if sample.get('proto', 0) <= 0.7846153974533081:
                                                                                    return 2
                                                                                else:
                                                                                    if sample.get('proto', 0) <= 0.8846153914928436:
                                                                                        return 3
                                                                                    else:
                                                                                        return 2
                                                                            else:
                                                                                return 2
                                                                        else:
                                                                            if sample.get('ct_srv_src', 0) <= 0.10483870655298233:
                                                                                if sample.get('ct_src_ltm', 0) <= 0.16101694852113724:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.042372881434857845:
                                                                                        return 3
                                                                                    else:
                                                                                        return 2
                                                                                else:
                                                                                    return 2
                                                                            else:
                                                                                return 2
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                        else:
                                            if sample.get('ct_srv_src', 0) <= 0.5403226017951965:
                                                return 6
                                            else:
                                                return 3
                            else:
                                if sample.get('sbytes', 0) <= 6.408581839423277e-06:
                                    if sample.get('dmean', 0) <= 0.029333332553505898:
                                        return 5
                                    else:
                                        return 3
                                else:
                                    if sample.get('sbytes', 0) <= 8.916287697502412e-06:
                                        return 3
                                    else:
                                        return 3
                    else:
                        if sample.get('ct_srv_dst', 0) <= 0.008196720853447914:
                            if sample.get('sbytes', 0) <= 3.08587150357198e-05:
                                if sample.get('state', 0) <= 0.6666666567325592:
                                    return 3
                                else:
                                    return 3
                            else:
                                if sample.get('dur', 0) <= 0.014128577429801226:
                                    if sample.get('sbytes', 0) <= 5.558748307521455e-05:
                                        if sample.get('sjit', 0) <= 0.003530218033120036:
                                            if sample.get('service', 0) <= 0.0833333358168602:
                                                if sample.get('sbytes', 0) <= 3.657071283669211e-05:
                                                    return 3
                                                else:
                                                    if sample.get('dloss', 0) <= 0.0002723806173889898:
                                                        return 8
                                                    else:
                                                        return 3
                                            else:
                                                if sample.get('stcpb', 0) <= 0.4106011688709259:
                                                    if sample.get('dinpkt', 0) <= 0.0013693199143745005:
                                                        return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('dur', 0) <= 0.00438783410936594:
                                                        return 3
                                                    else:
                                                        return 2
                                        else:
                                            if sample.get('sinpkt', 0) <= 0.0019023588974960148:
                                                return 7
                                            else:
                                                return 3
                                    else:
                                        if sample.get('spkts', 0) <= 0.004603100125677884:
                                            if sample.get('rate', 0) <= 1.9011992662854027e-05:
                                                return 3
                                            else:
                                                if sample.get('dmean', 0) <= 0.09733333066105843:
                                                    if sample.get('spkts', 0) <= 0.001784875465091318:
                                                        if sample.get('smean', 0) <= 0.050337838008999825:
                                                            if sample.get('stcpb', 0) <= 0.635134756565094:
                                                                if sample.get('dmean', 0) <= 0.04499999899417162:
                                                                    if sample.get('dtcpb', 0) <= 0.07418356090784073:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.05932203307747841:
                                                                            if sample.get('stcpb', 0) <= 0.37290313839912415:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('dinpkt', 0) <= 0.00219199457205832:
                                                                                    return 7
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('synack', 0) <= 0.04202801734209061:
                                                                if sample.get('dtcpb', 0) <= 0.30436915159225464:
                                                                    if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                                                        return 2
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('dtcpb', 0) <= 0.874814510345459:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('synack', 0) <= 0.018825842067599297:
                                                                            return 2
                                                                        else:
                                                                            return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('spkts', 0) <= 0.0025364019675180316:
                                                            return 3
                                                        else:
                                                            if sample.get('sloss', 0) <= 0.002914081560447812:
                                                                return 3
                                                            else:
                                                                return 3
                                                else:
                                                    return 3
                                        else:
                                            if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                return 6
                                            else:
                                                return 3
                                else:
                                    if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                        if sample.get('service', 0) <= 0.5833333283662796:
                                            if sample.get('sbytes', 0) <= 4.005363734904677e-05:
                                                if sample.get('sinpkt', 0) <= 0.002855094033293426:
                                                    return 7
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('dload', 0) <= 0.0004116704367334023:
                                                    if sample.get('dbytes', 0) <= 3.104206280113431e-05:
                                                        if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                                            return 6
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('synack', 0) <= 0.04735281504690647:
                                                            return 6
                                                        else:
                                                            return 3
                                                else:
                                                    return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sbytes', 0) <= 5.482123924593907e-05:
                                            if sample.get('dmean', 0) <= 0.029666665941476822:
                                                if sample.get('smean', 0) <= 0.029729729518294334:
                                                    return 7
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('synack', 0) <= 0.012475254014134407:
                                                    return 3
                                                else:
                                                    if sample.get('sload', 0) <= 3.45091478948234e-07:
                                                        return 3
                                                    else:
                                                        if sample.get('ct_src_ltm', 0) <= 0.31355932354927063:
                                                            return 6
                                                        else:
                                                            return 3
                                        else:
                                            if sample.get('dtcpb', 0) <= 0.7381693720817566:
                                                if sample.get('smean', 0) <= 0.04999999888241291:
                                                    if sample.get('sinpkt', 0) <= 0.0018207654356956482:
                                                        if sample.get('dinpkt', 0) <= 0.002430422813631594:
                                                            return 4
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('rate', 0) <= 1.200677525048377e-05:
                                                        if sample.get('synack', 0) <= 0.014230405446141958:
                                                            return 3
                                                        else:
                                                            return 6
                                                    else:
                                                        if sample.get('dtcpb', 0) <= 0.4767865538597107:
                                                            return 3
                                                        else:
                                                            if sample.get('smean', 0) <= 0.06081080995500088:
                                                                return 3
                                                            else:
                                                                if sample.get('service', 0) <= 0.5:
                                                                    return 4
                                                                else:
                                                                    return 3
                                            else:
                                                if sample.get('rate', 0) <= 8.253581427197787e-06:
                                                    return 3
                                                else:
                                                    return 6
                        else:
                            if sample.get('ct_dst_src_ltm', 0) <= 0.024193547666072845:
                                if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                    if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                        if sample.get('tcprtt', 0) <= 0.018058650195598602:
                                            if sample.get('smean', 0) <= 0.021283783949911594:
                                                if sample.get('state', 0) <= 0.25:
                                                    return 3
                                                else:
                                                    return 6
                                            else:
                                                if sample.get('sinpkt', 0) <= 0.0007628679159097373:
                                                    if sample.get('stcpb', 0) <= 0.2236875742673874:
                                                        if sample.get('stcpb', 0) <= 0.18422600626945496:
                                                            return 3
                                                        else:
                                                            return 7
                                                    else:
                                                        if sample.get('stcpb', 0) <= 0.7534689009189606:
                                                            if sample.get('tcprtt', 0) <= 0.016252798028290272:
                                                                return 3
                                                            else:
                                                                if sample.get('dinpkt', 0) <= 0.0007651417108718306:
                                                                    return 3
                                                                else:
                                                                    return 3
                                                        else:
                                                            if sample.get('smean', 0) <= 0.11993243172764778:
                                                                if sample.get('synack', 0) <= 0.0023205738980323076:
                                                                    return 3
                                                                else:
                                                                    return 2
                                                            else:
                                                                return 3
                                                else:
                                                    if sample.get('sbytes', 0) <= 4.569597695081029e-05:
                                                        return 7
                                                    else:
                                                        return 3
                                        else:
                                            if sample.get('sloss', 0) <= 0.0004700131685240194:
                                                if sample.get('rate', 0) <= 5.044931640441064e-05:
                                                    if sample.get('dinpkt', 0) <= 0.007536530727520585:
                                                        if sample.get('dloss', 0) <= 0.00045396767382044345:
                                                            if sample.get('synack', 0) <= 0.006433797301724553:
                                                                return 3
                                                            else:
                                                                if sample.get('sjit', 0) <= 0.0005737869723816402:
                                                                    return 3
                                                                else:
                                                                    if sample.get('ct_srv_src', 0) <= 0.15322580188512802:
                                                                        if sample.get('sbytes', 0) <= 3.503822517814115e-05:
                                                                            if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                                                                return 6
                                                                            else:
                                                                                return 4
                                                                        else:
                                                                            if sample.get('rate', 0) <= 3.6378231015987694e-05:
                                                                                if sample.get('sbytes', 0) <= 5.307977698976174e-05:
                                                                                    if sample.get('trans_depth', 0) <= 0.003816793905571103:
                                                                                        if sample.get('tcprtt', 0) <= 0.0849391557276249:
                                                                                            if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                                                                                if sample.get('stcpb', 0) <= 0.03582172095775604:
                                                                                                    if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                                        return 6
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    if sample.get('sinpkt', 0) <= 0.002935881493613124:
                                                                                                        if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                                                            if sample.get('smean', 0) <= 0.033445945009589195:
                                                                                                                if sample.get('dtcpb', 0) <= 0.1365528106689453:
                                                                                                                    if sample.get('dtcpb', 0) <= 0.1329687535762787:
                                                                                                                        return 6
                                                                                                                    else:
                                                                                                                        return 3
                                                                                                                else:
                                                                                                                    return 6
                                                                                                            else:
                                                                                                                if sample.get('stcpb', 0) <= 0.8121797144412994:
                                                                                                                    return 6
                                                                                                                else:
                                                                                                                    if sample.get('smean', 0) <= 0.03445945866405964:
                                                                                                                        return 3
                                                                                                                    else:
                                                                                                                        return 6
                                                                                                        else:
                                                                                                            if sample.get('dur', 0) <= 0.019076828844845295:
                                                                                                                return 6
                                                                                                            else:
                                                                                                                if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                                                                                    return 3
                                                                                                                else:
                                                                                                                    return 6
                                                                                                    else:
                                                                                                        if sample.get('dinpkt', 0) <= 0.0037215406773611903:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 6
                                                                                            else:
                                                                                                if sample.get('dtcpb', 0) <= 0.7202886044979095:
                                                                                                    return 6
                                                                                                else:
                                                                                                    if sample.get('dtcpb', 0) <= 0.8535219728946686:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        if sample.get('stcpb', 0) <= 0.5236869752407074:
                                                                                                            return 4
                                                                                                        else:
                                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('dur', 0) <= 0.023829503916203976:
                                                                                                return 6
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        if sample.get('sinpkt', 0) <= 0.0018157133599743247:
                                                                                            return 2
                                                                                        else:
                                                                                            return 4
                                                                                else:
                                                                                    if sample.get('ackdat', 0) <= 0.019484917633235455:
                                                                                        if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                                                                            if sample.get('ackdat', 0) <= 0.018832257017493248:
                                                                                                return 6
                                                                                            else:
                                                                                                if sample.get('ackdat', 0) <= 0.01890720333904028:
                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('dtcpb', 0) <= 0.6842688322067261:
                                                                                                        return 6
                                                                                                    else:
                                                                                                        return 3
                                                                                        else:
                                                                                            if sample.get('sbytes', 0) <= 0.00011904637722182088:
                                                                                                return 3
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 6
                                                                            else:
                                                                                if sample.get('dur', 0) <= 0.00740249315276742:
                                                                                    return 6
                                                                                else:
                                                                                    return 3
                                                                    else:
                                                                        return 3
                                                        else:
                                                            if sample.get('ackdat', 0) <= 0.05573723837733269:
                                                                if sample.get('ackdat', 0) <= 0.014083518646657467:
                                                                    if sample.get('ackdat', 0) <= 0.013157876674085855:
                                                                        if sample.get('dtcpb', 0) <= 0.22815927863121033:
                                                                            return 3
                                                                        else:
                                                                            return 6
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('tcprtt', 0) <= 0.04923308081924915:
                                                                        if sample.get('ackdat', 0) <= 0.02822627779096365:
                                                                            return 6
                                                                        else:
                                                                            if sample.get('ackdat', 0) <= 0.028979492373764515:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                    else:
                                                                        if sample.get('dtcpb', 0) <= 0.8106998801231384:
                                                                            if sample.get('tcprtt', 0) <= 0.05029170773923397:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                        else:
                                                                            return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('dmean', 0) <= 0.038333332166075706:
                                                    if sample.get('smean', 0) <= 0.0179054057225585:
                                                        if sample.get('rate', 0) <= 1.1336782336002216e-05:
                                                            return 3
                                                        else:
                                                            if sample.get('ackdat', 0) <= 0.045253003016114235:
                                                                return 6
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('stcpb', 0) <= 0.004829037934541702:
                                                            return 3
                                                        else:
                                                            if sample.get('dload', 0) <= 0.00036171634565107524:
                                                                if sample.get('sbytes', 0) <= 5.3567386203212664e-05:
                                                                    if sample.get('sbytes', 0) <= 5.314943518897053e-05:
                                                                        if sample.get('tcprtt', 0) <= 0.04606191627681255:
                                                                            if sample.get('stcpb', 0) <= 0.037060558795928955:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('sbytes', 0) <= 4.2422025217092596e-05:
                                                                                    if sample.get('tcprtt', 0) <= 0.029798518866300583:
                                                                                        return 3
                                                                                    else:
                                                                                        return 6
                                                                                else:
                                                                                    return 6
                                                                        else:
                                                                            if sample.get('dur', 0) <= 0.01492333598434925:
                                                                                return 4
                                                                            else:
                                                                                if sample.get('dtcpb', 0) <= 0.32621626555919647:
                                                                                    if sample.get('stcpb', 0) <= 0.6248077750205994:
                                                                                        return 6
                                                                                    else:
                                                                                        if sample.get('stcpb', 0) <= 0.8109828233718872:
                                                                                            return 4
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    return 6
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('dur', 0) <= 0.009399818256497383:
                                                                        if sample.get('djit', 0) <= 0.00032807607203722:
                                                                            return 6
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 6
                                                            else:
                                                                if sample.get('sjit', 0) <= 0.0019336878904141486:
                                                                    return 6
                                                                else:
                                                                    return 3
                                                else:
                                                    if sample.get('ackdat', 0) <= 0.043048158288002014:
                                                        if sample.get('sbytes', 0) <= 9.821848288993351e-05:
                                                            if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                if sample.get('sjit', 0) <= 0.007097103400155902:
                                                                    if sample.get('dtcpb', 0) <= 0.013705100398510695:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('synack', 0) <= 0.06420719064772129:
                                                                            if sample.get('dtcpb', 0) <= 0.8293204605579376:
                                                                                if sample.get('ackdat', 0) <= 0.04127882048487663:
                                                                                    if sample.get('dmean', 0) <= 0.06300000101327896:
                                                                                        return 6
                                                                                    else:
                                                                                        if sample.get('tcprtt', 0) <= 0.04343007132411003:
                                                                                            if sample.get('djit', 0) <= 0.00023848973796702921:
                                                                                                return 3
                                                                                            else:
                                                                                                return 4
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('dtcpb', 0) <= 0.8779100477695465:
                                                                                    return 3
                                                                                else:
                                                                                    return 6
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('tcprtt', 0) <= 0.05580752342939377:
                                                                        return 4
                                                                    else:
                                                                        return 3
                                                            else:
                                                                if sample.get('dtcpb', 0) <= 0.295034795999527:
                                                                    return 3
                                                                else:
                                                                    if sample.get('stcpb', 0) <= 0.7131058275699615:
                                                                        if sample.get('sbytes', 0) <= 7.077303234837018e-05:
                                                                            if sample.get('smean', 0) <= 0.0405405405908823:
                                                                                return 3
                                                                            else:
                                                                                return 4
                                                                        else:
                                                                            return 6
                                                                    else:
                                                                        return 3
                                                        else:
                                                            if sample.get('dinpkt', 0) <= 0.003993898630142212:
                                                                if sample.get('djit', 0) <= 0.00026679080474423245:
                                                                    return 3
                                                                else:
                                                                    if sample.get('tcprtt', 0) <= 0.0665319450199604:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                            return 4
                                                                        else:
                                                                            return 6
                                                                    else:
                                                                        return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('synack', 0) <= 0.04123450443148613:
                                                            if sample.get('stcpb', 0) <= 0.45421281456947327:
                                                                return 6
                                                            else:
                                                                return 4
                                                        else:
                                                            return 4
                                    else:
                                        if sample.get('sbytes', 0) <= 3.315744470455684e-05:
                                            if sample.get('sbytes', 0) <= 1.4349650882650167e-05:
                                                if sample.get('rate', 0) <= 0.05556177072958235:
                                                    return 3
                                                else:
                                                    return 6
                                            else:
                                                return 3
                                        else:
                                            if sample.get('synack', 0) <= 0.00806126743555069:
                                                if sample.get('dload', 0) <= 0.00010157479482586496:
                                                    if sample.get('smean', 0) <= 0.019256756640970707:
                                                        return 3
                                                    else:
                                                        return 4
                                                else:
                                                    if sample.get('ct_dst_ltm', 0) <= 0.14655172359198332:
                                                        if sample.get('stcpb', 0) <= 0.9512637555599213:
                                                            if sample.get('dtcpb', 0) <= 0.9804853796958923:
                                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                    return 3
                                                                else:
                                                                    if sample.get('dload', 0) <= 0.00023139894619816914:
                                                                        if sample.get('dinpkt', 0) <= 0.0017542526475153863:
                                                                            return 7
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('synack', 0) <= 0.0018820573459379375:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('dloss', 0) <= 0.0007263482839334756:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('dmean', 0) <= 0.04533333331346512:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 5
                                                    else:
                                                        return 6
                                            else:
                                                if sample.get('sinpkt', 0) <= 0.029212876223027706:
                                                    if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                        if sample.get('synack', 0) <= 0.014463764149695635:
                                                            if sample.get('djit', 0) <= 0.0012168965768069029:
                                                                if sample.get('dloss', 0) <= 0.00045396767382044345:
                                                                    if sample.get('smean', 0) <= 0.041554052382707596:
                                                                        if sample.get('dload', 0) <= 0.001196127908769995:
                                                                            return 6
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('dload', 0) <= 0.0004762345488416031:
                                                                        return 4
                                                                    else:
                                                                        return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('smean', 0) <= 0.7645270228385925:
                                                                if sample.get('dload', 0) <= 0.0005765896057710052:
                                                                    if sample.get('dloss', 0) <= 0.00063555475207977:
                                                                        if sample.get('synack', 0) <= 0.015221948269754648:
                                                                            if sample.get('sload', 0) <= 2.8922769388373126e-06:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('sbytes', 0) <= 4.0750221160124056e-05:
                                                                                if sample.get('sbytes', 0) <= 3.754593126359396e-05:
                                                                                    if sample.get('service', 0) <= 0.2083333283662796:
                                                                                        if sample.get('stcpb', 0) <= 0.7156628966331482:
                                                                                            return 6
                                                                                        else:
                                                                                            if sample.get('stcpb', 0) <= 0.7287247478961945:
                                                                                                return 3
                                                                                            else:
                                                                                                return 6
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    if sample.get('sbytes', 0) <= 3.7824565879418515e-05:
                                                                                        return 7
                                                                                    else:
                                                                                        if sample.get('sinpkt', 0) <= 0.0019011880503967404:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('sload', 0) <= 6.894938735513279e-07:
                                                                                                return 6
                                                                                            else:
                                                                                                return 3
                                                                            else:
                                                                                if sample.get('stcpb', 0) <= 0.0052206190302968025:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                                        if sample.get('synack', 0) <= 0.02251805830746889:
                                                                                            return 3
                                                                                        else:
                                                                                            return 6
                                                                                    else:
                                                                                        if sample.get('stcpb', 0) <= 0.904631108045578:
                                                                                            return 6
                                                                                        else:
                                                                                            if sample.get('dbytes', 0) <= 5.546636748476885e-05:
                                                                                                return 6
                                                                                            else:
                                                                                                return 3
                                                                    else:
                                                                        if sample.get('sbytes', 0) <= 0.00012030022662656847:
                                                                            if sample.get('stcpb', 0) <= 0.19924776256084442:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('dload', 0) <= 0.0005961519491393119:
                                                                        if sample.get('dtcpb', 0) <= 0.7447284758090973:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('ct_flw_http_mthd', 0) <= 0.03125:
                                                                            return 3
                                                                        else:
                                                                            return 6
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('dbytes', 0) <= 4.530094338406343e-05:
                                                            if sample.get('tcprtt', 0) <= 0.025529345497488976:
                                                                if sample.get('service', 0) <= 0.2083333283662796:
                                                                    return 4
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                                                    if sample.get('dload', 0) <= 0.0002585487673059106:
                                                                        if sample.get('dtcpb', 0) <= 0.04721381887793541:
                                                                            if sample.get('sjit', 0) <= 0.0022183757973834872:
                                                                                return 4
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('dtcpb', 0) <= 0.9566110074520111:
                                                                                if sample.get('sload', 0) <= 1.9912610227379446e-07:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('ackdat', 0) <= 0.019578472711145878:
                                                                                        if sample.get('dtcpb', 0) <= 0.7876646816730499:
                                                                                            if sample.get('ackdat', 0) <= 0.019113773480057716:
                                                                                                if sample.get('stcpb', 0) <= 0.1790311485528946:
                                                                                                    if sample.get('smean', 0) <= 0.02060810849070549:
                                                                                                        return 8
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    return 6
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            return 4
                                                                                    else:
                                                                                        if sample.get('tcprtt', 0) <= 0.029896780848503113:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('smean', 0) <= 0.0179054057225585:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                                                                                    return 6
                                                                                                else:
                                                                                                    if sample.get('ackdat', 0) <= 0.04430072754621506:
                                                                                                        if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                                            return 6
                                                                                                        else:
                                                                                                            return 3
                                                                                                    else:
                                                                                                        return 4
                                                                            else:
                                                                                if sample.get('dmean', 0) <= 0.028999999165534973:
                                                                                    return 3
                                                                                else:
                                                                                    return 4
                                                                    else:
                                                                        if sample.get('dinpkt', 0) <= 0.0010033983853645623:
                                                                            if sample.get('sinpkt', 0) <= 0.0007452527352143079:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 6
                                                                else:
                                                                    if sample.get('synack', 0) <= 0.028466387651860714:
                                                                        if sample.get('synack', 0) <= 0.012917954474687576:
                                                                            return 3
                                                                        else:
                                                                            return 6
                                                                    else:
                                                                        if sample.get('dload', 0) <= 0.0001377714506816119:
                                                                            return 4
                                                                        else:
                                                                            return 3
                                                        else:
                                                            if sample.get('dbytes', 0) <= 5.887758197786752e-05:
                                                                if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                                                    if sample.get('dmean', 0) <= 0.052666667848825455:
                                                                        if sample.get('sloss', 0) <= 0.0009400263370480388:
                                                                            return 6
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('sload', 0) <= 2.638416617628536e-06:
                                                                            if sample.get('djit', 0) <= 0.0009329754684586078:
                                                                                return 4
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('sjit', 0) <= 0.007255819160491228:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                            return 3
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        return 6
                                                            else:
                                                                if sample.get('dinpkt', 0) <= 0.0012329149176366627:
                                                                    return 3
                                                                else:
                                                                    if sample.get('rate', 0) <= 7.388098674709909e-05:
                                                                        return 6
                                                                    else:
                                                                        return 3
                                                else:
                                                    if sample.get('sbytes', 0) <= 8.317224637721665e-05:
                                                        if sample.get('ct_srv_dst', 0) <= 0.024590162560343742:
                                                            if sample.get('dmean', 0) <= 0.04833333380520344:
                                                                return 6
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('smean', 0) <= 0.024324324913322926:
                                                                return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        return 4
                                else:
                                    if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                        if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                            if sample.get('dload', 0) <= 0.00023908273578854278:
                                                if sample.get('tcprtt', 0) <= 0.022455262951552868:
                                                    return 3
                                                else:
                                                    if sample.get('ct_dst_ltm', 0) <= 0.043103449046611786:
                                                        if sample.get('stcpb', 0) <= 0.023291129618883133:
                                                            return 3
                                                        else:
                                                            if sample.get('ct_src_ltm', 0) <= 0.08474576286971569:
                                                                if sample.get('ackdat', 0) <= 0.02083650603890419:
                                                                    if sample.get('tcprtt', 0) <= 0.03592130355536938:
                                                                        return 6
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.5154250115156174:
                                                                            return 3
                                                                        else:
                                                                            return 6
                                                                else:
                                                                    if sample.get('sload', 0) <= 6.848071052445448e-06:
                                                                        return 6
                                                                    else:
                                                                        if sample.get('sloss', 0) <= 0.0009400263079442084:
                                                                            return 3
                                                                        else:
                                                                            return 6
                                                            else:
                                                                return 3
                                                    else:
                                                        return 3
                                            else:
                                                if sample.get('dtcpb', 0) <= 0.4541207402944565:
                                                    return 3
                                                else:
                                                    if sample.get('dload', 0) <= 0.0005845557316206396:
                                                        return 3
                                                    else:
                                                        return 3
                                        else:
                                            if sample.get('sbytes', 0) <= 6.387684334185906e-05:
                                                if sample.get('synack', 0) <= 0.02313941903412342:
                                                    if sample.get('dloss', 0) <= 0.0002723806173889898:
                                                        return 7
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('dload', 0) <= 0.00020410997240105644:
                                                        return 4
                                                    else:
                                                        return 3
                                            else:
                                                if sample.get('sinpkt', 0) <= 0.0016715755336917937:
                                                    if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                        return 6
                                                    else:
                                                        if sample.get('djit', 0) <= 0.0002795537147903815:
                                                            return 3
                                                        else:
                                                            if sample.get('synack', 0) <= 0.015755760483443737:
                                                                return 3
                                                            else:
                                                                if sample.get('sjit', 0) <= 0.0024042451987043023:
                                                                    return 3
                                                                else:
                                                                    return 8
                                                else:
                                                    return 6
                                    else:
                                        if sample.get('sttl', 0) <= 0.6196078434586525:
                                            if sample.get('djit', 0) <= 0.00013623714039567858:
                                                if sample.get('spkts', 0) <= 0.0006575857405550778:
                                                    return 6
                                                else:
                                                    return 3
                                            else:
                                                return 0
                                        else:
                                            if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                return 3
                                            else:
                                                if sample.get('dtcpb', 0) <= 0.002366960165090859:
                                                    if sample.get('ct_srv_dst', 0) <= 0.09836065024137497:
                                                        return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('stcpb', 0) <= 0.06111088767647743:
                                                        if sample.get('djit', 0) <= 0.0005749519041273743:
                                                            if sample.get('stcpb', 0) <= 0.05881689302623272:
                                                                return 4
                                                            else:
                                                                return 3
                                                        else:
                                                            return 6
                                                    else:
                                                        if sample.get('sjit', 0) <= 0.0032322799088433385:
                                                            if sample.get('stcpb', 0) <= 0.30828024446964264:
                                                                if sample.get('stcpb', 0) <= 0.3036460876464844:
                                                                    if sample.get('dur', 0) <= 0.006184634519740939:
                                                                        if sample.get('rate', 0) <= 4.1458495616097935e-05:
                                                                            return 6
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        return 4
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 4
                                                        else:
                                                            if sample.get('sjit', 0) <= 0.003304335754364729:
                                                                return 3
                                                            else:
                                                                if sample.get('dinpkt', 0) <= 0.0013793371035717428:
                                                                    return 3
                                                                else:
                                                                    if sample.get('djit', 0) <= 0.0004751871747430414:
                                                                        if sample.get('rate', 0) <= 1.7143388504337054e-05:
                                                                            return 3
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        return 4
                            else:
                                if sample.get('dload', 0) <= 0.00021917754202149808:
                                    if sample.get('service', 0) <= 0.125:
                                        if sample.get('djit', 0) <= 0.00020848612621193752:
                                            return 6
                                        else:
                                            if sample.get('rate', 0) <= 1.2327174658821605e-06:
                                                if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                                    if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                                        return 6
                                                    else:
                                                        return 4
                                                else:
                                                    if sample.get('djit', 0) <= 0.014275745954364538:
                                                        return 3
                                                    else:
                                                        if sample.get('dur', 0) <= 0.551910400390625:
                                                            return 3
                                                        else:
                                                            return 6
                                            else:
                                                if sample.get('smean', 0) <= 0.01587837841361761:
                                                    return 3
                                                else:
                                                    if sample.get('dmean', 0) <= 0.038333332166075706:
                                                        if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                            return 3
                                                        else:
                                                            if sample.get('ct_srv_dst', 0) <= 0.13934426009655:
                                                                if sample.get('ackdat', 0) <= 0.0071237904485315084:
                                                                    return 3
                                                                else:
                                                                    if sample.get('dloss', 0) <= 0.0013619030360132456:
                                                                        if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                                            if sample.get('sjit', 0) <= 0.000546597468201071:
                                                                                if sample.get('sbytes', 0) <= 0.00034028176742140204:
                                                                                    return 4
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('sbytes', 0) <= 3.27394936903147e-05:
                                                                                    if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                                                                        return 6
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    if sample.get('synack', 0) <= 0.08001021295785904:
                                                                                        if sample.get('ackdat', 0) <= 0.07916868478059769:
                                                                                            if sample.get('rate', 0) <= 6.1501020809373586e-06:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('smean', 0) <= 0.020608107559382915:
                                                                                                    if sample.get('dload', 0) <= 0.00020604328892659396:
                                                                                                        if sample.get('ct_srv_src', 0) <= 0.15322580188512802:
                                                                                                            if sample.get('dinpkt', 0) <= 0.001263641519472003:
                                                                                                                if sample.get('dinpkt', 0) <= 0.001260453078430146:
                                                                                                                    return 6
                                                                                                                else:
                                                                                                                    return 3
                                                                                                            else:
                                                                                                                if sample.get('sinpkt', 0) <= 0.0008519053517375141:
                                                                                                                    if sample.get('sinpkt', 0) <= 0.0008472978079225868:
                                                                                                                        return 6
                                                                                                                    else:
                                                                                                                        return 4
                                                                                                                else:
                                                                                                                    if sample.get('ackdat', 0) <= 0.01559233944863081:
                                                                                                                        if sample.get('ackdat', 0) <= 0.015582608059048653:
                                                                                                                            return 6
                                                                                                                        else:
                                                                                                                            return 3
                                                                                                                    else:
                                                                                                                        return 6
                                                                                                        else:
                                                                                                            if sample.get('synack', 0) <= 0.01869103219360113:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                return 6
                                                                                                    else:
                                                                                                        if sample.get('ct_srv_src', 0) <= 0.06451612710952759:
                                                                                                            return 6
                                                                                                        else:
                                                                                                            return 4
                                                                                                else:
                                                                                                    if sample.get('dur', 0) <= 0.006864484399557114:
                                                                                                        if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 6
                                                                                                    else:
                                                                                                        if sample.get('ackdat', 0) <= 0.024215389974415302:
                                                                                                            if sample.get('stcpb', 0) <= 0.005076027242466807:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                if sample.get('dtcpb', 0) <= 0.3938267230987549:
                                                                                                                    if sample.get('dtcpb', 0) <= 0.39326272904872894:
                                                                                                                        if sample.get('tcprtt', 0) <= 0.053685955703258514:
                                                                                                                            if sample.get('synack', 0) <= 0.013604705687612295:
                                                                                                                                if sample.get('sinpkt', 0) <= 0.0013961860677227378:
                                                                                                                                    return 6
                                                                                                                                else:
                                                                                                                                    return 3
                                                                                                                            else:
                                                                                                                                if sample.get('dload', 0) <= 0.00010067759649246:
                                                                                                                                    if sample.get('rate', 0) <= 1.9608414731919765e-05:
                                                                                                                                        if sample.get('sjit', 0) <= 0.0036938535049557686:
                                                                                                                                            return 3
                                                                                                                                        else:
                                                                                                                                            return 6
                                                                                                                                    else:
                                                                                                                                        return 3
                                                                                                                                else:
                                                                                                                                    if sample.get('dtcpb', 0) <= 0.3549978584051132:
                                                                                                                                        return 6
                                                                                                                                    else:
                                                                                                                                        if sample.get('dtcpb', 0) <= 0.3571935445070267:
                                                                                                                                            return 3
                                                                                                                                        else:
                                                                                                                                            return 6
                                                                                                                        else:
                                                                                                                            return 3
                                                                                                                    else:
                                                                                                                        return 3
                                                                                                                else:
                                                                                                                    return 6
                                                                                                        else:
                                                                                                            if sample.get('ackdat', 0) <= 0.02422546222805977:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                if sample.get('ct_dst_ltm', 0) <= 0.043103449046611786:
                                                                                                                    if sample.get('sbytes', 0) <= 3.8451491491287015e-05:
                                                                                                                        if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                                                                                                            if sample.get('stcpb', 0) <= 0.4035451263189316:
                                                                                                                                if sample.get('sjit', 0) <= 0.0038275219267234206:
                                                                                                                                    return 3
                                                                                                                                else:
                                                                                                                                    if sample.get('dur', 0) <= 0.01964843738824129:
                                                                                                                                        return 6
                                                                                                                                    else:
                                                                                                                                        return 3
                                                                                                                            else:
                                                                                                                                if sample.get('synack', 0) <= 0.022179950959980488:
                                                                                                                                    return 3
                                                                                                                                else:
                                                                                                                                    return 6
                                                                                                                        else:
                                                                                                                            if sample.get('tcprtt', 0) <= 0.03754594177007675:
                                                                                                                                return 3
                                                                                                                            else:
                                                                                                                                return 6
                                                                                                                    else:
                                                                                                                        if sample.get('ackdat', 0) <= 0.025428523309528828:
                                                                                                                            if sample.get('synack', 0) <= 0.046170370653271675:
                                                                                                                                if sample.get('ackdat', 0) <= 0.025399159640073776:
                                                                                                                                    if sample.get('sinpkt', 0) <= 0.0023297666339203715:
                                                                                                                                        if sample.get('dtcpb', 0) <= 0.9494891464710236:
                                                                                                                                            return 6
                                                                                                                                        else:
                                                                                                                                            return 3
                                                                                                                                    else:
                                                                                                                                        return 3
                                                                                                                                else:
                                                                                                                                    return 3
                                                                                                                            else:
                                                                                                                                return 3
                                                                                                                        else:
                                                                                                                            if sample.get('sinpkt', 0) <= 0.0014825068064965308:
                                                                                                                                if sample.get('sinpkt', 0) <= 0.001480578794144094:
                                                                                                                                    if sample.get('djit', 0) <= 0.0005339510971680284:
                                                                                                                                        if sample.get('dur', 0) <= 0.007886193227022886:
                                                                                                                                            if sample.get('sinpkt', 0) <= 0.0008452332986053079:
                                                                                                                                                return 6
                                                                                                                                            else:
                                                                                                                                                return 3
                                                                                                                                        else:
                                                                                                                                            if sample.get('stcpb', 0) <= 0.009709471371024847:
                                                                                                                                                return 3
                                                                                                                                            else:
                                                                                                                                                if sample.get('stcpb', 0) <= 0.9687502384185791:
                                                                                                                                                    if sample.get('smean', 0) <= 0.022635134868323803:
                                                                                                                                                        if sample.get('synack', 0) <= 0.0413862019777298:
                                                                                                                                                            return 6
                                                                                                                                                        else:
                                                                                                                                                            return 3
                                                                                                                                                    else:
                                                                                                                                                        if sample.get('rate', 0) <= 2.406511612207396e-05:
                                                                                                                                                            if sample.get('sjit', 0) <= 0.0022572583984583616:
                                                                                                                                                                return 3
                                                                                                                                                            else:
                                                                                                                                                                if sample.get('dtcpb', 0) <= 0.9325909614562988:
                                                                                                                                                                    if sample.get('djit', 0) <= 0.0003242190578021109:
                                                                                                                                                                        if sample.get('dinpkt', 0) <= 0.001822838094085455:
                                                                                                                                                                            return 6
                                                                                                                                                                        else:
                                                                                                                                                                            return 3
                                                                                                                                                                    else:
                                                                                                                                                                        if sample.get('sbytes', 0) <= 3.935705353796948e-05:
                                                                                                                                                                            if sample.get('ct_srv_src', 0) <= 0.09677419066429138:
                                                                                                                                                                                return 6
                                                                                                                                                                            else:
                                                                                                                                                                                return 3
                                                                                                                                                                        else:
                                                                                                                                                                            return 6
                                                                                                                                                                else:
                                                                                                                                                                    if sample.get('sinpkt', 0) <= 0.0012008666526526213:
                                                                                                                                                                        return 4
                                                                                                                                                                    else:
                                                                                                                                                                        return 6
                                                                                                                                                        else:
                                                                                                                                                            return 6
                                                                                                                                                else:
                                                                                                                                                    return 3
                                                                                                                                    else:
                                                                                                                                        if sample.get('stcpb', 0) <= 0.5510948896408081:
                                                                                                                                            return 6
                                                                                                                                        else:
                                                                                                                                            return 3
                                                                                                                                else:
                                                                                                                                    return 3
                                                                                                                            else:
                                                                                                                                if sample.get('synack', 0) <= 0.015531233977526426:
                                                                                                                                    return 3
                                                                                                                                else:
                                                                                                                                    if sample.get('ackdat', 0) <= 0.02660085540264845:
                                                                                                                                        if sample.get('ackdat', 0) <= 0.026560907252132893:
                                                                                                                                            return 6
                                                                                                                                        else:
                                                                                                                                            return 3
                                                                                                                                    else:
                                                                                                                                        return 6
                                                                                                                else:
                                                                                                                    if sample.get('ct_src_ltm', 0) <= 0.04237288050353527:
                                                                                                                        return 6
                                                                                                                    else:
                                                                                                                        return 4
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 3
                                                                        else:
                                                                            if sample.get('sbytes', 0) <= 8.533165964763612e-05:
                                                                                if sample.get('ct_srv_src', 0) <= 0.25806451588869095:
                                                                                    if sample.get('synack', 0) <= 0.04723396897315979:
                                                                                        if sample.get('tcprtt', 0) <= 0.020228891633450985:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('ct_dst_ltm', 0) <= 0.07758620753884315:
                                                                                                if sample.get('dur', 0) <= 0.008186076302081347:
                                                                                                    if sample.get('djit', 0) <= 0.00030382593104150146:
                                                                                                        if sample.get('sinpkt', 0) <= 0.0009279262740164995:
                                                                                                            if sample.get('dtcpb', 0) <= 0.024772795848548412:
                                                                                                                return 4
                                                                                                            else:
                                                                                                                if sample.get('ackdat', 0) <= 0.03310476243495941:
                                                                                                                    if sample.get('sjit', 0) <= 0.0014471493195742369:
                                                                                                                        if sample.get('tcprtt', 0) <= 0.029086227528750896:
                                                                                                                            return 6
                                                                                                                        else:
                                                                                                                            return 4
                                                                                                                    else:
                                                                                                                        if sample.get('dtcpb', 0) <= 0.32455000281333923:
                                                                                                                            if sample.get('dtcpb', 0) <= 0.31463976204395294:
                                                                                                                                if sample.get('dtcpb', 0) <= 0.20913614332675934:
                                                                                                                                    return 6
                                                                                                                                else:
                                                                                                                                    if sample.get('sjit', 0) <= 0.0015918475110083818:
                                                                                                                                        return 3
                                                                                                                                    else:
                                                                                                                                        return 6
                                                                                                                            else:
                                                                                                                                return 3
                                                                                                                        else:
                                                                                                                            return 6
                                                                                                                else:
                                                                                                                    return 3
                                                                                                        else:
                                                                                                            return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                                                                                        if sample.get('sinpkt', 0) <= 0.0033744501415640116:
                                                                                                            if sample.get('ct_srv_src', 0) <= 0.08064515888690948:
                                                                                                                if sample.get('rate', 0) <= 1.0951329386443831e-05:
                                                                                                                    return 3
                                                                                                                else:
                                                                                                                    return 6
                                                                                                            else:
                                                                                                                if sample.get('dload', 0) <= 0.00017705000936985016:
                                                                                                                    if sample.get('smean', 0) <= 0.03547297231853008:
                                                                                                                        return 3
                                                                                                                    else:
                                                                                                                        return 6
                                                                                                                else:
                                                                                                                    return 3
                                                                                                        else:
                                                                                                            return 3
                                                                                                    else:
                                                                                                        if sample.get('synack', 0) <= 0.01270969770848751:
                                                                                                            if sample.get('dbytes', 0) <= 2.1217761968728155e-05:
                                                                                                                return 6
                                                                                                            else:
                                                                                                                return 3
                                                                                                        else:
                                                                                                            if sample.get('stcpb', 0) <= 0.09996937960386276:
                                                                                                                if sample.get('stcpb', 0) <= 0.09663153439760208:
                                                                                                                    if sample.get('djit', 0) <= 0.00028125487733632326:
                                                                                                                        return 3
                                                                                                                    else:
                                                                                                                        return 6
                                                                                                                else:
                                                                                                                    return 3
                                                                                                            else:
                                                                                                                if sample.get('dtcpb', 0) <= 0.9806406199932098:
                                                                                                                    if sample.get('ackdat', 0) <= 0.04310176521539688:
                                                                                                                        if sample.get('dtcpb', 0) <= 0.011372117791324854:
                                                                                                                            return 3
                                                                                                                        else:
                                                                                                                            if sample.get('sjit', 0) <= 0.002091742935590446:
                                                                                                                                if sample.get('sjit', 0) <= 0.0020887638675048947:
                                                                                                                                    if sample.get('djit', 0) <= 0.00032888051646295935:
                                                                                                                                        return 6
                                                                                                                                    else:
                                                                                                                                        if sample.get('djit', 0) <= 0.00033312586310785264:
                                                                                                                                            return 3
                                                                                                                                        else:
                                                                                                                                            return 6
                                                                                                                                else:
                                                                                                                                    return 3
                                                                                                                            else:
                                                                                                                                return 6
                                                                                                                    else:
                                                                                                                        if sample.get('ackdat', 0) <= 0.043718233704566956:
                                                                                                                            return 3
                                                                                                                        else:
                                                                                                                            if sample.get('tcprtt', 0) <= 0.05638374388217926:
                                                                                                                                return 3
                                                                                                                            else:
                                                                                                                                return 6
                                                                                                                else:
                                                                                                                    if sample.get('dtcpb', 0) <= 0.982594221830368:
                                                                                                                        return 3
                                                                                                                    else:
                                                                                                                        return 6
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        if sample.get('ackdat', 0) <= 0.022921846248209476:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('sbytes', 0) <= 3.3436079320381396e-05:
                                                                                                return 3
                                                                                            else:
                                                                                                return 6
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('sbytes', 0) <= 9.598941323929466e-05:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('djit', 0) <= 0.07665028795599937:
                                                                                        if sample.get('ackdat', 0) <= 0.009132307022809982:
                                                                                            return 4
                                                                                        else:
                                                                                            if sample.get('ackdat', 0) <= 0.027579762041568756:
                                                                                                if sample.get('dtcpb', 0) <= 0.009321873541921377:
                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('dtcpb', 0) <= 0.994985818862915:
                                                                                                        if sample.get('stcpb', 0) <= 0.8756695091724396:
                                                                                                            if sample.get('stcpb', 0) <= 0.035189525224268436:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                if sample.get('synack', 0) <= 0.025760446675121784:
                                                                                                                    return 6
                                                                                                                else:
                                                                                                                    if sample.get('synack', 0) <= 0.02608615718781948:
                                                                                                                        return 3
                                                                                                                    else:
                                                                                                                        if sample.get('dur', 0) <= 0.008062101434916258:
                                                                                                                            if sample.get('sload', 0) <= 5.591974058916094e-06:
                                                                                                                                return 6
                                                                                                                            else:
                                                                                                                                return 3
                                                                                                                        else:
                                                                                                                            return 6
                                                                                                        else:
                                                                                                            if sample.get('sjit', 0) <= 0.002654079464264214:
                                                                                                                if sample.get('dinpkt', 0) <= 0.0012811668566428125:
                                                                                                                    return 6
                                                                                                                else:
                                                                                                                    if sample.get('djit', 0) <= 0.00034425040939822793:
                                                                                                                        return 4
                                                                                                                    else:
                                                                                                                        return 3
                                                                                                            else:
                                                                                                                return 6
                                                                                                    else:
                                                                                                        return 3
                                                                                            else:
                                                                                                if sample.get('synack', 0) <= 0.024764563888311386:
                                                                                                    if sample.get('ackdat', 0) <= 0.03107746597379446:
                                                                                                        if sample.get('smean', 0) <= 0.10270270332694054:
                                                                                                            return 6
                                                                                                        else:
                                                                                                            return 4
                                                                                                    else:
                                                                                                        if sample.get('dload', 0) <= 0.00012105201676604338:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            if sample.get('djit', 0) <= 0.0002482779527781531:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                return 6
                                                                                                else:
                                                                                                    if sample.get('dur', 0) <= 0.007356526330113411:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        if sample.get('ackdat', 0) <= 0.02762944158166647:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            if sample.get('sjit', 0) <= 0.0022590288426727057:
                                                                                                                if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                                                                    if sample.get('synack', 0) <= 0.045836446806788445:
                                                                                                                        return 6
                                                                                                                    else:
                                                                                                                        return 3
                                                                                                                else:
                                                                                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                                                                                                                        return 3
                                                                                                                    else:
                                                                                                                        return 6
                                                                                                            else:
                                                                                                                return 6
                                                                                    else:
                                                                                        return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                if sample.get('ct_dst_src_ltm', 0) <= 0.08064515888690948:
                                                                    if sample.get('dload', 0) <= 9.678346395958215e-05:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                            return 3
                                                                        else:
                                                                            return 4
                                                                else:
                                                                    if sample.get('sload', 0) <= 5.853017455592635e-06:
                                                                        if sample.get('dload', 0) <= 0.00021616350568365306:
                                                                            return 6
                                                                        else:
                                                                            if sample.get('ackdat', 0) <= 0.017451134975999594:
                                                                                return 4
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 3
                                                    else:
                                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                                            if sample.get('djit', 0) <= 0.0004558252840070054:
                                                                if sample.get('dur', 0) <= 0.022715353406965733:
                                                                    if sample.get('synack', 0) <= 0.028041042387485504:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 4
                                                            else:
                                                                if sample.get('dbytes', 0) <= 4.2367299101897515e-05:
                                                                    return 3
                                                                else:
                                                                    if sample.get('synack', 0) <= 0.017337365075945854:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('dbytes', 0) <= 5.034954483562615e-05:
                                                                                if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                                                                    if sample.get('dmean', 0) <= 0.060999998822808266:
                                                                                        return 6
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 6
                                                        else:
                                                            if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                                                return 4
                                                            else:
                                                                return 3
                                    else:
                                        if sample.get('dload', 0) <= 0.00014630209625465795:
                                            if sample.get('stcpb', 0) <= 0.730787605047226:
                                                if sample.get('ct_src_dport_ltm', 0) <= 0.025862068869173527:
                                                    if sample.get('sjit', 0) <= 0.0021253350423648953:
                                                        return 3
                                                    else:
                                                        if sample.get('ackdat', 0) <= 0.015320382546633482:
                                                            return 3
                                                        else:
                                                            if sample.get('dbytes', 0) <= 4.946262743033003e-05:
                                                                return 4
                                                            else:
                                                                if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                                                    return 4
                                                                else:
                                                                    return 3
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('sbytes', 0) <= 4.0123295548255555e-05:
                                                    return 4
                                                else:
                                                    return 6
                                        else:
                                            if sample.get('ct_src_ltm', 0) <= 0.0338983042165637:
                                                if sample.get('sloss', 0) <= 0.007332205539569259:
                                                    if sample.get('stcpb', 0) <= 0.042800876311957836:
                                                        return 3
                                                    else:
                                                        if sample.get('dinpkt', 0) <= 0.001181279367301613:
                                                            return 3
                                                        else:
                                                            if sample.get('sbytes', 0) <= 3.719763844856061e-05:
                                                                return 3
                                                            else:
                                                                return 6
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('dpkts', 0) <= 0.000635324016911909:
                                                    return 3
                                                else:
                                                    if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                                        return 3
                                                    else:
                                                        if sample.get('ct_srv_src', 0) <= 0.04838709533214569:
                                                            return 4
                                                        else:
                                                            return 3
                                else:
                                    if sample.get('ct_srv_dst', 0) <= 0.0901639349758625:
                                        if sample.get('response_body_len', 0) <= 2.117156964231981e-05:
                                            if sample.get('rate', 0) <= 2.3659783437324222e-05:
                                                if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                                    if sample.get('sbytes', 0) <= 6.840464629931375e-05:
                                                        if sample.get('sbytes', 0) <= 5.314943518897053e-05:
                                                            if sample.get('sjit', 0) <= 0.00392809952609241:
                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 6
                                                    else:
                                                        if sample.get('ct_srv_src', 0) <= 0.07258064299821854:
                                                            if sample.get('ct_dst_ltm', 0) <= 0.008620689623057842:
                                                                if sample.get('dpkts', 0) <= 0.000998366333078593:
                                                                    if sample.get('stcpb', 0) <= 0.7214363813400269:
                                                                        if sample.get('sjit', 0) <= 0.005223040701821446:
                                                                            return 4
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 6
                                                else:
                                                    if sample.get('sloss', 0) <= 0.0003760105391847901:
                                                        return 3
                                                    else:
                                                        return 4
                                            else:
                                                if sample.get('sttl', 0) <= 0.6196078434586525:
                                                    return 3
                                                else:
                                                    if sample.get('sbytes', 0) <= 3.21125680784462e-05:
                                                        if sample.get('smean', 0) <= 0.015540540684014559:
                                                            return 2
                                                        else:
                                                            return 4
                                                    else:
                                                        if sample.get('ct_dst_src_ltm', 0) <= 0.10483870655298233:
                                                            if sample.get('dinpkt', 0) <= 0.0007833603594917804:
                                                                if sample.get('dpkts', 0) <= 0.0013614085619337857:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.04237288050353527:
                                                                        if sample.get('ackdat', 0) <= 0.0030529798241332173:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('dtcpb', 0) <= 0.027969016693532467:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('rate', 0) <= 9.234610843122937e-05:
                                                                                    if sample.get('synack', 0) <= 0.01243775524199009:
                                                                                        if sample.get('dinpkt', 0) <= 0.0007284285093192011:
                                                                                            if sample.get('dur', 0) <= 0.00884301820769906:
                                                                                                return 6
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            if sample.get('dload', 0) <= 0.00034058817254845053:
                                                                                                return 3
                                                                                            else:
                                                                                                return 4
                                                                                    else:
                                                                                        return 6
                                                                                else:
                                                                                    if sample.get('stcpb', 0) <= 0.2700885683298111:
                                                                                        return 3
                                                                                    else:
                                                                                        return 6
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.6946655511856079:
                                                                            if sample.get('rate', 0) <= 6.785456935176626e-05:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('synack', 0) <= 0.014522026292979717:
                                                                        return 3
                                                                    else:
                                                                        return 4
                                                            else:
                                                                if sample.get('dmean', 0) <= 0.06833333522081375:
                                                                    if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                                                        if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('dmean', 0) <= 0.0409999992698431:
                                                                                if sample.get('service', 0) <= 0.125:
                                                                                    if sample.get('dbytes', 0) <= 6.044674228178337e-05:
                                                                                        if sample.get('ackdat', 0) <= 0.041910482570528984:
                                                                                            if sample.get('dload', 0) <= 0.00022941130009712651:
                                                                                                if sample.get('ackdat', 0) <= 0.00789680890738964:
                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('synack', 0) <= 0.026233517564833164:
                                                                                                        if sample.get('dinpkt', 0) <= 0.001161474036052823:
                                                                                                            if sample.get('sbytes', 0) <= 8.35205428302288e-05:
                                                                                                                if sample.get('ackdat', 0) <= 0.009177718311548233:
                                                                                                                    return 3
                                                                                                                else:
                                                                                                                    return 6
                                                                                                            else:
                                                                                                                return 3
                                                                                                        else:
                                                                                                            return 6
                                                                                                    else:
                                                                                                        if sample.get('tcprtt', 0) <= 0.03839338943362236:
                                                                                                            if sample.get('sjit', 0) <= 0.001512513612397015:
                                                                                                                return 6
                                                                                                            else:
                                                                                                                return 4
                                                                                                        else:
                                                                                                            return 6
                                                                                            else:
                                                                                                if sample.get('dtcpb', 0) <= 0.9755432903766632:
                                                                                                    if sample.get('sinpkt', 0) <= 0.0011149771162308753:
                                                                                                        if sample.get('dtcpb', 0) <= 0.13090000301599503:
                                                                                                            if sample.get('sinpkt', 0) <= 0.0003406115429243073:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                if sample.get('dmean', 0) <= 0.038333332166075706:
                                                                                                                    if sample.get('rate', 0) <= 5.091714047011919e-05:
                                                                                                                        return 6
                                                                                                                    else:
                                                                                                                        if sample.get('dinpkt', 0) <= 0.0011625614133663476:
                                                                                                                            return 6
                                                                                                                        else:
                                                                                                                            return 3
                                                                                                                else:
                                                                                                                    return 3
                                                                                                        else:
                                                                                                            if sample.get('tcprtt', 0) <= 0.01713976263999939:
                                                                                                                if sample.get('tcprtt', 0) <= 0.0170897813513875:
                                                                                                                    return 6
                                                                                                                else:
                                                                                                                    return 3
                                                                                                            else:
                                                                                                                if sample.get('dmean', 0) <= 0.029666665941476822:
                                                                                                                    if sample.get('dur', 0) <= 0.006854301318526268:
                                                                                                                        return 3
                                                                                                                    else:
                                                                                                                        return 6
                                                                                                                else:
                                                                                                                    if sample.get('sinpkt', 0) <= 0.0003844203893095255:
                                                                                                                        if sample.get('sinpkt', 0) <= 0.00038304159534163773:
                                                                                                                            return 6
                                                                                                                        else:
                                                                                                                            return 3
                                                                                                                    else:
                                                                                                                        return 6
                                                                                                    else:
                                                                                                        if sample.get('djit', 0) <= 0.00019961388170486316:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            if sample.get('sinpkt', 0) <= 0.00111713248770684:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                if sample.get('djit', 0) <= 0.00022116804757388309:
                                                                                                                    return 3
                                                                                                                else:
                                                                                                                    return 6
                                                                                                else:
                                                                                                    if sample.get('dinpkt', 0) <= 0.0008896520303096622:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        if sample.get('sjit', 0) <= 0.0024490224895998836:
                                                                                                            return 6
                                                                                                        else:
                                                                                                            return 4
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('synack', 0) <= 0.017664005048573017:
                                                                                            return 6
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    if sample.get('spkts', 0) <= 0.001127289782743901:
                                                                                        if sample.get('synack', 0) <= 0.016324748285114765:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('dtcpb', 0) <= 0.12728125974535942:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('dur', 0) <= 0.005618650931864977:
                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('stcpb', 0) <= 0.6379299759864807:
                                                                                                        if sample.get('spkts', 0) <= 0.0009394081716891378:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 6
                                                                                                    else:
                                                                                                        return 4
                                                                                    else:
                                                                                        return 6
                                                                            else:
                                                                                if sample.get('smean', 0) <= 0.06790540367364883:
                                                                                    if sample.get('stcpb', 0) <= 0.9544300734996796:
                                                                                        if sample.get('sinpkt', 0) <= 0.001163400651421398:
                                                                                            if sample.get('dinpkt', 0) <= 0.00152563740266487:
                                                                                                return 6
                                                                                            else:
                                                                                                if sample.get('dload', 0) <= 0.00036689723492600024:
                                                                                                    return 6
                                                                                                else:
                                                                                                    return 3
                                                                                        else:
                                                                                            if sample.get('rate', 0) <= 2.560064513090765e-05:
                                                                                                return 6
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    if sample.get('rate', 0) <= 4.259011438989546e-05:
                                                                                        return 4
                                                                                    else:
                                                                                        return 3
                                                                    else:
                                                                        if sample.get('djit', 0) <= 0.000155604888277594:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('synack', 0) <= 0.012810572050511837:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('dur', 0) <= 0.016394236125051975:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.04237288050353527:
                                                                                        return 6
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                else:
                                                                    return 3
                                                        else:
                                                            return 3
                                        else:
                                            if sample.get('smean', 0) <= 0.045270269736647606:
                                                if sample.get('ackdat', 0) <= 0.005355646600946784:
                                                    return 3
                                                else:
                                                    if sample.get('sload', 0) <= 3.706916800183535e-06:
                                                        if sample.get('dload', 0) <= 0.0009619885531719774:
                                                            if sample.get('ct_src_ltm', 0) <= 0.04237288050353527:
                                                                if sample.get('dtcpb', 0) <= 0.6742910444736481:
                                                                    return 6
                                                                else:
                                                                    if sample.get('dtcpb', 0) <= 0.6757550239562988:
                                                                        return 3
                                                                    else:
                                                                        return 6
                                                            else:
                                                                if sample.get('dbytes', 0) <= 9.530936586088501e-05:
                                                                    if sample.get('synack', 0) <= 0.0178047027438879:
                                                                        if sample.get('stcpb', 0) <= 0.6053856313228607:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        return 6
                                                                else:
                                                                    return 3
                                                        else:
                                                            if sample.get('stcpb', 0) <= 0.053833987563848495:
                                                                return 3
                                                            else:
                                                                if sample.get('dload', 0) <= 0.000965274462942034:
                                                                    return 3
                                                                else:
                                                                    return 6
                                                    else:
                                                        return 3
                                            else:
                                                return 3
                                    else:
                                        if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                                            if sample.get('dur', 0) <= 0.010543360374867916:
                                                if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                    return 3
                                                else:
                                                    if sample.get('dmean', 0) <= 0.09266666695475578:
                                                        return 4
                                                    else:
                                                        return 3
                                            else:
                                                if sample.get('dload', 0) <= 0.0004325476475059986:
                                                    return 6
                                                else:
                                                    return 3
                                        else:
                                            if sample.get('sbytes', 0) <= 6.157811367302202e-05:
                                                if sample.get('dloss', 0) <= 0.0008171418448910117:
                                                    if sample.get('ct_srv_src', 0) <= 0.16935483366250992:
                                                        if sample.get('ackdat', 0) <= 0.016741624101996422:
                                                            if sample.get('stcpb', 0) <= 0.8585203886032104:
                                                                return 6
                                                            else:
                                                                if sample.get('stcpb', 0) <= 0.8613739609718323:
                                                                    return 3
                                                                else:
                                                                    return 6
                                                        else:
                                                            if sample.get('ackdat', 0) <= 0.016911489889025688:
                                                                return 3
                                                            else:
                                                                if sample.get('synack', 0) <= 0.005424279486760497:
                                                                    return 3
                                                                else:
                                                                    if sample.get('dload', 0) <= 0.00029472820460796356:
                                                                        if sample.get('dload', 0) <= 0.0002770584251265973:
                                                                            if sample.get('dtcpb', 0) <= 0.07773912325501442:
                                                                                return 4
                                                                            else:
                                                                                if sample.get('dtcpb', 0) <= 0.7920979857444763:
                                                                                    return 6
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            if sample.get('dur', 0) <= 0.006941692903637886:
                                                                                return 3
                                                                            else:
                                                                                return 6
                                                                    else:
                                                                        return 6
                                                    else:
                                                        if sample.get('service', 0) <= 0.2083333283662796:
                                                            return 6
                                                        else:
                                                            return 4
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('rate', 0) <= 3.228810601285659e-05:
                                                    return 6
                                                else:
                                                    if sample.get('sload', 0) <= 4.406159632708295e-06:
                                                        if sample.get('dur', 0) <= 0.009863101877272129:
                                                            return 4
                                                        else:
                                                            if sample.get('dbytes', 0) <= 4.680188067140989e-05:
                                                                return 6
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('dinpkt', 0) <= 0.0010993590694852173:
                                                            if sample.get('stcpb', 0) <= 0.7434457838535309:
                                                                if sample.get('sload', 0) <= 1.0880451100092614e-05:
                                                                    if sample.get('dload', 0) <= 0.00022879428433952853:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.21568801254034042:
                                                                            if sample.get('stcpb', 0) <= 0.14994213730096817:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 6
                                                                else:
                                                                    if sample.get('ct_srv_src', 0) <= 0.08870967477560043:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('sbytes', 0) <= 0.0012017484405077994:
                                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.13709677010774612:
                                                                                return 6
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 4
                                                            else:
                                                                if sample.get('dload', 0) <= 0.00035249000939074904:
                                                                    if sample.get('rate', 0) <= 3.591225504351314e-05:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.940515786409378:
                                                                            return 4
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    return 3
                                                        else:
                                                            return 4
            else:
                if sample.get('sttl', 0) <= 0.6215686276555061:
                    if sample.get('sloss', 0) <= 0.0010340289736632258:
                        if sample.get('sbytes', 0) <= 6.93798647262156e-05:
                            if sample.get('djit', 0) <= 0.00018019463459495455:
                                if sample.get('dtcpb', 0) <= 0.9266005456447601:
                                    if sample.get('ackdat', 0) <= 0.02644533012062311:
                                        if sample.get('smean', 0) <= 0.04932432435452938:
                                            if sample.get('dload', 0) <= 0.02016426809132099:
                                                if sample.get('dmean', 0) <= 0.23200000077486038:
                                                    if sample.get('dmean', 0) <= 0.1756666675209999:
                                                        if sample.get('dload', 0) <= 0.0025076245656237006:
                                                            if sample.get('djit', 0) <= 0.00011360757707734592:
                                                                return 3
                                                            else:
                                                                if sample.get('tcprtt', 0) <= 0.01474552322179079:
                                                                    if sample.get('dinpkt', 0) <= 0.0006217806658241898:
                                                                        if sample.get('tcprtt', 0) <= 0.013262453023344278:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('response_body_len', 0) <= 9.765625145519152e-05:
                                                                                return 2
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        if sample.get('ackdat', 0) <= 0.015273264143615961:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('ackdat', 0) <= 0.008229712024331093:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dload', 0) <= 0.002450105734169483:
                                                                            if sample.get('stcpb', 0) <= 0.18596740812063217:
                                                                                if sample.get('stcpb', 0) <= 0.12423131987452507:
                                                                                    return 3
                                                                                else:
                                                                                    return 2
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('synack', 0) <= 0.002017331193201244:
                                                            return 3
                                                        else:
                                                            if sample.get('tcprtt', 0) <= 0.01574239181354642:
                                                                if sample.get('smean', 0) <= 0.03682432509958744:
                                                                    return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('tcprtt', 0) <= 0.017837530933320522:
                                                                    return 2
                                                                else:
                                                                    if sample.get('ackdat', 0) <= 0.01711242739111185:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                else:
                                                    if sample.get('stcpb', 0) <= 0.9238664209842682:
                                                        return 3
                                                    else:
                                                        return 3
                                            else:
                                                return 2
                                        else:
                                            return 3
                                    else:
                                        return 3
                                else:
                                    if sample.get('dtcpb', 0) <= 0.9605186283588409:
                                        return 3
                                    else:
                                        if sample.get('dmean', 0) <= 0.13599999994039536:
                                            return 3
                                        else:
                                            return 3
                            else:
                                if sample.get('dbytes', 0) <= 0.0038389822002500296:
                                    if sample.get('ackdat', 0) <= 0.05486810579895973:
                                        if sample.get('smean', 0) <= 0.016554053872823715:
                                            return 3
                                        else:
                                            if sample.get('smean', 0) <= 0.019256756640970707:
                                                return 3
                                            else:
                                                if sample.get('dbytes', 0) <= 0.00014961591659812257:
                                                    if sample.get('response_body_len', 0) <= 0.00011749267650884576:
                                                        if sample.get('smean', 0) <= 0.04797297343611717:
                                                            if sample.get('dtcpb', 0) <= 0.9703612923622131:
                                                                if sample.get('sbytes', 0) <= 5.677167609974276e-05:
                                                                    if sample.get('ackdat', 0) <= 0.012907943222671747:
                                                                        if sample.get('dmean', 0) <= 0.12799999862909317:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('dbytes', 0) <= 0.00010363273395341821:
                                                                            if sample.get('synack', 0) <= 0.026122571900486946:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('sbytes', 0) <= 5.691099249816034e-05:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('dbytes', 0) <= 0.0001379495588480495:
                                                                            if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('service', 0) <= 0.2083333283662796:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            if sample.get('stcpb', 0) <= 0.44524794816970825:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                                else:
                                                    if sample.get('smean', 0) <= 0.02871621586382389:
                                                        if sample.get('sbytes', 0) <= 5.5796457672840916e-05:
                                                            return 5
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('sbytes', 0) <= 5.7468261729809456e-05:
                                                            if sample.get('smean', 0) <= 0.03141891770064831:
                                                                return 3
                                                            else:
                                                                if sample.get('synack', 0) <= 0.025093219242990017:
                                                                    if sample.get('stcpb', 0) <= 0.011492060963064432:
                                                                        if sample.get('ackdat', 0) <= 0.012153021059930325:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('synack', 0) <= 0.002612969954498112:
                                                                            if sample.get('dbytes', 0) <= 0.00024628976825624704:
                                                                                if sample.get('sinpkt', 0) <= 0.00035253110399935395:
                                                                                    return 3
                                                                                else:
                                                                                    return 2
                                                                            else:
                                                                                if sample.get('synack', 0) <= 0.0025275908410549164:
                                                                                    if sample.get('rate', 0) <= 0.00010990422379109077:
                                                                                        if sample.get('stcpb', 0) <= 0.9626771211624146:
                                                                                            if sample.get('tcprtt', 0) <= 0.014943614136427641:
                                                                                                return 3
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('rate', 0) <= 0.00012020073700114153:
                                                                                            if sample.get('dload', 0) <= 0.013171609956771135:
                                                                                                return 3
                                                                                            else:
                                                                                                return 2
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    if sample.get('sinpkt', 0) <= 0.00036078065750189126:
                                                                                        return 3
                                                                                    else:
                                                                                        return 2
                                                                        else:
                                                                            if sample.get('djit', 0) <= 0.01641455665230751:
                                                                                if sample.get('rate', 0) <= 3.9179445593617857e-05:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('dload', 0) <= 0.002296719467267394:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('tcprtt', 0) <= 0.036643145605921745:
                                                                                            if sample.get('sbytes', 0) <= 5.391567901824601e-05:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('sbytes', 0) <= 5.426397001428995e-05:
                                                                                                    if sample.get('ackdat', 0) <= 0.013703326229006052:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                                else:
                                                                                                    if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                                                        if sample.get('tcprtt', 0) <= 0.010305602569133043:
                                                                                                            if sample.get('djit', 0) <= 0.0037823771126568317:
                                                                                                                return 3
                                                                                                            else:
                                                                                                                return 3
                                                                                                        else:
                                                                                                            if sample.get('sbytes', 0) <= 5.454260463011451e-05:
                                                                                                                if sample.get('stcpb', 0) <= 0.1851474568247795:
                                                                                                                    return 3
                                                                                                                else:
                                                                                                                    return 3
                                                                                                            else:
                                                                                                                return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                        else:
                                                                                            if sample.get('ackdat', 0) <= 0.032055518589913845:
                                                                                                return 2
                                                                                            else:
                                                                                                return 3
                                                                            else:
                                                                                if sample.get('stcpb', 0) <= 0.9021270871162415:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                                                        return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                else:
                                                                    if sample.get('synack', 0) <= 0.028216449543833733:
                                                                        if sample.get('synack', 0) <= 0.027663422748446465:
                                                                            if sample.get('synack', 0) <= 0.025257470086216927:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('dbytes', 0) <= 0.000507725344505161:
                                                                                    if sample.get('dload', 0) <= 0.000842245586682111:
                                                                                        return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('sbytes', 0) <= 5.71896289329743e-05:
                                                                            if sample.get('dload', 0) <= 0.0037020727759227157:
                                                                                if sample.get('dtcpb', 0) <= 0.9176824390888214:
                                                                                    if sample.get('dload', 0) <= 0.0006857551925349981:
                                                                                        return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    if sample.get('ct_srv_src', 0) <= 0.016129031777381897:
                                                                                        return 3
                                                                                    else:
                                                                                        return 3
                                                                            else:
                                                                                if sample.get('dload', 0) <= 0.004579626256600022:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 3
                                                        else:
                                                            if sample.get('dtcpb', 0) <= 0.00496737111825496:
                                                                return 3
                                                            else:
                                                                if sample.get('stcpb', 0) <= 0.9966676533222198:
                                                                    if sample.get('dmean', 0) <= 0.1653333306312561:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.856233149766922:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('stcpb', 0) <= 0.8586896657943726:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                else:
                                                                    return 3
                                    else:
                                        if sample.get('dtcpb', 0) <= 0.37286175787448883:
                                            return 2
                                        else:
                                            if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                                return 3
                                            else:
                                                return 3
                                else:
                                    if sample.get('ackdat', 0) <= 0.007473424077033997:
                                        return 3
                                    else:
                                        return 3
                        else:
                            if sample.get('spkts', 0) <= 0.0009394081716891378:
                                if sample.get('sbytes', 0) <= 0.00010866726006497629:
                                    if sample.get('smean', 0) <= 0.061486486345529556:
                                        if sample.get('djit', 0) <= 0.008806865429505706:
                                            if sample.get('smean', 0) <= 0.059797296300530434:
                                                return 3
                                            else:
                                                return 3
                                        else:
                                            return 3
                                    else:
                                        if sample.get('dur', 0) <= 0.027105246670544147:
                                            if sample.get('stcpb', 0) <= 0.9923073053359985:
                                                return 5
                                            else:
                                                return 3
                                        else:
                                            return 3
                                else:
                                    return 3
                            else:
                                if sample.get('rate', 0) <= 0.0001054836466209963:
                                    if sample.get('smean', 0) <= 0.08209459483623505:
                                        if sample.get('sloss', 0) <= 0.0004700131685240194:
                                            if sample.get('smean', 0) <= 0.033445946872234344:
                                                if sample.get('stcpb', 0) <= 0.09958123788237572:
                                                    return 3
                                                else:
                                                    if sample.get('spkts', 0) <= 0.04509159363806248:
                                                        if sample.get('dinpkt', 0) <= 0.0032549953320994973:
                                                            if sample.get('sinpkt', 0) <= 0.0010550132137723267:
                                                                if sample.get('dmean', 0) <= 0.8303333222866058:
                                                                    return 3
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('ackdat', 0) <= 0.0124050029553473:
                                                                    return 3
                                                                else:
                                                                    return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                            else:
                                                if sample.get('tcprtt', 0) <= 0.018849576823413372:
                                                    if sample.get('dinpkt', 0) <= 0.00019900138431694359:
                                                        return 3
                                                    else:
                                                        return 2
                                                else:
                                                    if sample.get('sbytes', 0) <= 7.627605737070553e-05:
                                                        return 3
                                                    else:
                                                        if sample.get('stcpb', 0) <= 0.895541250705719:
                                                            if sample.get('ct_flw_http_mthd', 0) <= 0.15625:
                                                                return 5
                                                            else:
                                                                return 3
                                                        else:
                                                            return 3
                                        else:
                                            if sample.get('dur', 0) <= 0.0756578054279089:
                                                if sample.get('ct_src_dport_ltm', 0) <= 0.043103449046611786:
                                                    if sample.get('djit', 0) <= 0.0001141318389272783:
                                                        if sample.get('dload', 0) <= 0.0028097040485590696:
                                                            return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('dtcpb', 0) <= 0.037341016344726086:
                                                            return 3
                                                        else:
                                                            if sample.get('tcprtt', 0) <= 0.06483639031648636:
                                                                if sample.get('trans_depth', 0) <= 0.01145038171671331:
                                                                    return 3
                                                                else:
                                                                    if sample.get('sbytes', 0) <= 9.612872963771224e-05:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                return 3
                                                else:
                                                    if sample.get('dinpkt', 0) <= 0.0006921160966157913:
                                                        return 3
                                                    else:
                                                        return 6
                                            else:
                                                return 3
                                    else:
                                        if sample.get('djit', 0) <= 0.004214192042127252:
                                            if sample.get('sloss', 0) <= 0.0004700131685240194:
                                                return 3
                                            else:
                                                return 5
                                        else:
                                            return 3
                                else:
                                    if sample.get('smean', 0) <= 0.030743243172764778:
                                        if sample.get('dmean', 0) <= 0.9116666615009308:
                                            if sample.get('ct_src_ltm', 0) <= 0.025423728860914707:
                                                if sample.get('dur', 0) <= 0.3129556477069855:
                                                    if sample.get('sbytes', 0) <= 7.300210563698784e-05:
                                                        return 2
                                                    else:
                                                        if sample.get('ackdat', 0) <= 0.01497774850577116:
                                                            if sample.get('ct_src_dport_ltm', 0) <= 0.008620689623057842:
                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('tcprtt', 0) <= 0.013645814266055822:
                                                                if sample.get('sjit', 0) <= 0.0010040565393865108:
                                                                    return 3
                                                                else:
                                                                    return 2
                                                            else:
                                                                if sample.get('dmean', 0) <= 0.8806666731834412:
                                                                    if sample.get('ackdat', 0) <= 0.02168071549385786:
                                                                        if sample.get('dtcpb', 0) <= 0.1161828376352787:
                                                                            if sample.get('response_body_len', 0) <= 0.008827209239825606:
                                                                                return 3
                                                                            else:
                                                                                return 2
                                                                        else:
                                                                            if sample.get('smean', 0) <= 0.030067567713558674:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('sload', 0) <= 4.389088871903368e-06:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                    else:
                                                                        if sample.get('dtcpb', 0) <= 0.8920546770095825:
                                                                            if sample.get('dbytes', 0) <= 0.017912395764142275:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('synack', 0) <= 0.029744903557002544:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    return 3
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('dtcpb', 0) <= 0.08674854598939419:
                                                    return 3
                                                else:
                                                    if sample.get('sinpkt', 0) <= 0.000549776159459725:
                                                        return 3
                                                    else:
                                                        if sample.get('rate', 0) <= 0.0001108198775909841:
                                                            return 3
                                                        else:
                                                            return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('sloss', 0) <= 0.000658018427202478:
                                            if sample.get('dpkts', 0) <= 0.0019059720216318965:
                                                return 3
                                            else:
                                                if sample.get('ct_dst_src_ltm', 0) <= 0.016129031777381897:
                                                    if sample.get('dmean', 0) <= 0.8696666657924652:
                                                        if sample.get('sbytes', 0) <= 8.630688171251677e-05:
                                                            if sample.get('sinpkt', 0) <= 0.0004495834728004411:
                                                                return 5
                                                            else:
                                                                return 3
                                                        else:
                                                            return 5
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                        else:
                                            return 3
                    else:
                        if sample.get('dloss', 0) <= 0.0009987288794945925:
                            if sample.get('sbytes', 0) <= 0.060498056933283806:
                                if sample.get('smean', 0) <= 0.21250000596046448:
                                    if sample.get('ct_srv_src', 0) <= 0.14516128599643707:
                                        return 6
                                    else:
                                        return 3
                                else:
                                    return 2
                            else:
                                return 3
                        else:
                            if sample.get('dbytes', 0) <= 0.4879561513662338:
                                if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                    if sample.get('djit', 0) <= 0.00010119892976945266:
                                        if sample.get('sbytes', 0) <= 6.666318222414702e-05:
                                            return 3
                                        else:
                                            if sample.get('dload', 0) <= 0.0006863810995128006:
                                                if sample.get('dur', 0) <= 0.030234296806156635:
                                                    return 3
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('synack', 0) <= 0.0022327156038954854:
                                                    if sample.get('synack', 0) <= 0.0020886095007881522:
                                                        if sample.get('dinpkt', 0) <= 0.0005778689810540527:
                                                            return 3
                                                        else:
                                                            if sample.get('ackdat', 0) <= 0.0171578386798501:
                                                                return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('dinpkt', 0) <= 0.0006386128661688417:
                                                            return 2
                                                        else:
                                                            if sample.get('synack', 0) <= 0.002208078047260642:
                                                                return 3
                                                            else:
                                                                return 2
                                                else:
                                                    if sample.get('ackdat', 0) <= 0.01689100358635187:
                                                        if sample.get('synack', 0) <= 0.011176594067364931:
                                                            return 3
                                                        else:
                                                            if sample.get('synack', 0) <= 0.011241209227591753:
                                                                return 3
                                                            else:
                                                                return 3
                                                    else:
                                                        if sample.get('dur', 0) <= 0.010831468272954226:
                                                            return 3
                                                        else:
                                                            return 3
                                    else:
                                        if sample.get('dpkts', 0) <= 0.0016790705849416554:
                                            if sample.get('service', 0) <= 0.2083333283662796:
                                                if sample.get('stcpb', 0) <= 0.930147647857666:
                                                    if sample.get('ct_dst_src_ltm', 0) <= 0.06451612710952759:
                                                        if sample.get('ackdat', 0) <= 0.030979132279753685:
                                                            if sample.get('stcpb', 0) <= 0.852719247341156:
                                                                return 3
                                                            else:
                                                                if sample.get('stcpb', 0) <= 0.8645199239253998:
                                                                    return 3
                                                                else:
                                                                    return 3
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                            else:
                                                return 3
                                        else:
                                            if sample.get('ct_src_dport_ltm', 0) <= 0.034482759423553944:
                                                if sample.get('spkts', 0) <= 0.7299671173095703:
                                                    if sample.get('dmean', 0) <= 0.05766666494309902:
                                                        if sample.get('dbytes', 0) <= 0.00010718039629864506:
                                                            if sample.get('smean', 0) <= 0.3543918877840042:
                                                                return 3
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('ct_srv_src', 0) <= 0.008064515888690948:
                                                                if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                                                    if sample.get('stcpb', 0) <= 0.9050392806529999:
                                                                        return 3
                                                                    else:
                                                                        if sample.get('stcpb', 0) <= 0.9070850908756256:
                                                                            return 3
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('spkts', 0) <= 0.03372475365176797:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                            else:
                                                                if sample.get('stcpb', 0) <= 0.04900667257606983:
                                                                    if sample.get('stcpb', 0) <= 0.0429074652493:
                                                                        return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.11016948893666267:
                                                                        if sample.get('dpkts', 0) <= 0.0019059720216318965:
                                                                            if sample.get('ackdat', 0) <= 0.02617883589118719:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('synack', 0) <= 0.037949347868561745:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('tcprtt', 0) <= 0.050289351493120193:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                    else:
                                                                        return 3
                                                    else:
                                                        if sample.get('dload', 0) <= 7.05224483681377e-05:
                                                            return 3
                                                        else:
                                                            if sample.get('dtcpb', 0) <= 0.08529715240001678:
                                                                if sample.get('dtcpb', 0) <= 0.07792739570140839:
                                                                    if sample.get('dur', 0) <= 0.06164952926337719:
                                                                        if sample.get('stcpb', 0) <= 0.8991506397724152:
                                                                            if sample.get('synack', 0) <= 0.0020052448380738497:
                                                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('dload', 0) <= 0.03433789685368538:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('smean', 0) <= 0.016554053872823715:
                                                                                        return 3
                                                                                    else:
                                                                                        return 3
                                                                        else:
                                                                            if sample.get('spkts', 0) <= 0.0021606386872008443:
                                                                                return 3
                                                                            else:
                                                                                return 2
                                                                    else:
                                                                        if sample.get('sinpkt', 0) <= 0.002172622596845031:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('stcpb', 0) <= 0.16774806007742882:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                else:
                                                                    return 2
                                                            else:
                                                                if sample.get('dbytes', 0) <= 0.00012300843809498474:
                                                                    if sample.get('smean', 0) <= 0.07736486569046974:
                                                                        if sample.get('dtcpb', 0) <= 0.30654042959213257:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('sloss', 0) <= 0.0014100394328124821:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.05932203307747841:
                                                                            if sample.get('smean', 0) <= 0.08141892030835152:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('sjit', 0) <= 0.0028002363396808505:
                                                                                return 3
                                                                            else:
                                                                                return 3
                                                                else:
                                                                    if sample.get('dbytes', 0) <= 0.0003362094139447436:
                                                                        if sample.get('dbytes', 0) <= 0.0003160832420689985:
                                                                            if sample.get('dtcpb', 0) <= 0.2580167055130005:
                                                                                if sample.get('dtcpb', 0) <= 0.23906558007001877:
                                                                                    if sample.get('sload', 0) <= 9.85566884992295e-07:
                                                                                        if sample.get('sinpkt', 0) <= 0.0015654736198484898:
                                                                                            return 3
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('synack', 0) <= 0.0018712106393650174:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('stcpb', 0) <= 0.0760597288608551:
                                                                                        if sample.get('dload', 0) <= 0.0011152392253279686:
                                                                                            return 3
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('sloss', 0) <= 0.001598044706042856:
                                                                            if sample.get('spkts', 0) <= 0.0020666979253292084:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('ackdat', 0) <= 0.008684850763529539:
                                                                                    if sample.get('tcprtt', 0) <= 0.02322917990386486:
                                                                                        if sample.get('smean', 0) <= 0.014527027029544115:
                                                                                            if sample.get('sjit', 0) <= 0.0011440824600867927:
                                                                                                return 3
                                                                                            else:
                                                                                                return 2
                                                                                        else:
                                                                                            if sample.get('stcpb', 0) <= 0.22850176692008972:
                                                                                                if sample.get('stcpb', 0) <= 0.18187271803617477:
                                                                                                    return 3
                                                                                                else:
                                                                                                    return 3
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    if sample.get('sjit', 0) <= 0.006285786395892501:
                                                                                        if sample.get('dtcpb', 0) <= 0.7491782605648041:
                                                                                            if sample.get('smean', 0) <= 0.019256756640970707:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('sbytes', 0) <= 8.616756531409919e-05:
                                                                                                    return 2
                                                                                                else:
                                                                                                    if sample.get('dpkts', 0) <= 0.004991831490769982:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        return 3
                                                                                        else:
                                                                                            if sample.get('dtcpb', 0) <= 0.752752810716629:
                                                                                                return 3
                                                                                            else:
                                                                                                if sample.get('dtcpb', 0) <= 0.7570644915103912:
                                                                                                    return 3
                                                                                                else:
                                                                                                    if sample.get('sbytes', 0) <= 7.759956861264072e-05:
                                                                                                        return 3
                                                                                                    else:
                                                                                                        if sample.get('tcprtt', 0) <= 0.013495217077434063:
                                                                                                            return 3
                                                                                                        else:
                                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('sjit', 0) <= 0.0072997757233679295:
                                                                                            return 3
                                                                                        else:
                                                                                            return 3
                                                                        else:
                                                                            if sample.get('smean', 0) <= 0.06993243098258972:
                                                                                if sample.get('sinpkt', 0) <= 0.001827923348173499:
                                                                                    return 5
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                else:
                                                    return 3
                                            else:
                                                return 3
                                else:
                                    return 7
                            else:
                                return 2
                else:
                    if sample.get('dmean', 0) <= 0.06733333691954613:
                        if sample.get('smean', 0) <= 0.8158783614635468:
                            if sample.get('rate', 0) <= 4.972701390215661e-06:
                                return 4
                            else:
                                if sample.get('dpkts', 0) <= 0.0020874931942671537:
                                    if sample.get('smean', 0) <= 0.2037162110209465:
                                        if sample.get('sbytes', 0) <= 0.00014899953384883702:
                                            return 3
                                        else:
                                            if sample.get('tcprtt', 0) <= 0.04735199734568596:
                                                if sample.get('sload', 0) <= 6.29339615443314e-06:
                                                    if sample.get('stcpb', 0) <= 0.11442837864160538:
                                                        return 3
                                                    else:
                                                        return 2
                                                else:
                                                    return 3
                                            else:
                                                return 5
                                    else:
                                        if sample.get('sloss', 0) <= 0.0017860499792732298:
                                            return 5
                                        else:
                                            if sample.get('smean', 0) <= 0.38986486196517944:
                                                return 2
                                            else:
                                                if sample.get('smean', 0) <= 0.5385135114192963:
                                                    return 5
                                                else:
                                                    return 3
                                else:
                                    if sample.get('sjit', 0) <= 0.0005095089436508715:
                                        if sample.get('sjit', 0) <= 0.0002200896415160969:
                                            if sample.get('ct_src_ltm', 0) <= 0.15254237037152052:
                                                return 3
                                            else:
                                                return 2
                                        else:
                                            return 3
                                    else:
                                        if sample.get('ackdat', 0) <= 0.019116164185106754:
                                            return 2
                                        else:
                                            if sample.get('tcprtt', 0) <= 0.030531615018844604:
                                                return 3
                                            else:
                                                return 2
                        else:
                            if sample.get('spkts', 0) <= 0.02611554693430662:
                                if sample.get('dload', 0) <= 0.00032372269924962893:
                                    if sample.get('ct_srv_src', 0) <= 0.06451612710952759:
                                        return 6
                                    else:
                                        return 3
                                else:
                                    return 3
                            else:
                                if sample.get('ackdat', 0) <= 0.016473252326250076:
                                    if sample.get('sload', 0) <= 0.00018397328676655889:
                                        return 3
                                    else:
                                        return 3
                                else:
                                    if sample.get('dloss', 0) <= 0.0002723806173889898:
                                        return 2
                                    else:
                                        if sample.get('dinpkt', 0) <= 0.001057826040778309:
                                            return 2
                                        else:
                                            return 3
                    else:
                        if sample.get('djit', 0) <= 0.004139228956773877:
                            if sample.get('smean', 0) <= 0.0810810811817646:
                                if sample.get('dmean', 0) <= 0.17933333665132523:
                                    if sample.get('dbytes', 0) <= 0.00013747199409408495:
                                        if sample.get('sbytes', 0) <= 7.091235238476656e-05:
                                            if sample.get('smean', 0) <= 0.030743242241442204:
                                                return 3
                                            else:
                                                if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                                                    return 2
                                                else:
                                                    return 3
                                        else:
                                            if sample.get('service', 0) <= 0.2083333283662796:
                                                if sample.get('ct_src_ltm', 0) <= 0.008474576286971569:
                                                    if sample.get('stcpb', 0) <= 0.4791804552078247:
                                                        return 3
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                            else:
                                                return 5
                                    else:
                                        if sample.get('sbytes', 0) <= 0.0001799279052647762:
                                            if sample.get('stcpb', 0) <= 0.8861591517925262:
                                                if sample.get('djit', 0) <= 0.0010885213850997388:
                                                    return 2
                                                else:
                                                    return 3
                                            else:
                                                return 3
                                        else:
                                            if sample.get('dpkts', 0) <= 0.0019059720216318965:
                                                return 3
                                            else:
                                                if sample.get('dtcpb', 0) <= 0.8633202314376831:
                                                    return 2
                                                else:
                                                    return 3
                                else:
                                    if sample.get('sloss', 0) <= 0.00028200788801768795:
                                        return 2
                                    else:
                                        if sample.get('dpkts', 0) <= 0.0008168451604433358:
                                            return 3
                                        else:
                                            if sample.get('sload', 0) <= 1.7733298705024936e-06:
                                                if sample.get('djit', 0) <= 0.0021743387333117425:
                                                    return 3
                                                else:
                                                    return 9
                                            else:
                                                if sample.get('stcpb', 0) <= 0.05430133081972599:
                                                    return 3
                                                else:
                                                    return 3
                            else:
                                if sample.get('sjit', 0) <= 0.00010979377839248627:
                                    if sample.get('sload', 0) <= 1.190873263112735e-05:
                                        return 5
                                    else:
                                        return 2
                                else:
                                    if sample.get('service', 0) <= 0.875:
                                        if sample.get('ct_srv_src', 0) <= 0.08064515888690948:
                                            if sample.get('dbytes', 0) <= 0.0019330676877871156:
                                                if sample.get('ct_srv_dst', 0) <= 0.2295081913471222:
                                                    if sample.get('trans_depth', 0) <= 0.04580152686685324:
                                                        if sample.get('dbytes', 0) <= 0.00011004581756424159:
                                                            return 2
                                                        else:
                                                            if sample.get('ct_dst_sport_ltm', 0) <= 0.013513513840734959:
                                                                if sample.get('tcprtt', 0) <= 0.05858093686401844:
                                                                    if sample.get('ct_dst_ltm', 0) <= 0.034482759423553944:
                                                                        if sample.get('djit', 0) <= 0.0001105084411392454:
                                                                            if sample.get('dinpkt', 0) <= 0.0007950610888656229:
                                                                                if sample.get('response_body_len', 0) <= 0.00021276473853504285:
                                                                                    if sample.get('dbytes', 0) <= 0.00045287300599738955:
                                                                                        return 3
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            if sample.get('sbytes', 0) <= 0.00015728889411548153:
                                                                                if sample.get('stcpb', 0) <= 0.11274527385830879:
                                                                                    return 3
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                return 3
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    if sample.get('ct_srv_src', 0) <= 0.024193547666072845:
                                                                        return 3
                                                                    else:
                                                                        return 5
                                                            else:
                                                                return 3
                                                    else:
                                                        return 3
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('dload', 0) <= 0.010305034229531884:
                                                    return 1
                                                else:
                                                    return 3
                                        else:
                                            if sample.get('ackdat', 0) <= 0.018140842206776142:
                                                return 3
                                            else:
                                                if sample.get('dload', 0) <= 0.0009147586533799767:
                                                    return 3
                                                else:
                                                    return 3
                                    else:
                                        return 3
                        else:
                            if sample.get('smean', 0) <= 0.256418913602829:
                                if sample.get('dbytes', 0) <= 0.00036240756162442267:
                                    if sample.get('sbytes', 0) <= 5.969733319943771e-05:
                                        return 6
                                    else:
                                        if sample.get('djit', 0) <= 0.012140557635575533:
                                            return 3
                                        else:
                                            return 5
                                else:
                                    if sample.get('sbytes', 0) <= 7.25841564417351e-05:
                                        return 3
                                    else:
                                        if sample.get('dinpkt', 0) <= 0.00030840805266052485:
                                            return 3
                                        else:
                                            if sample.get('dtcpb', 0) <= 0.8877004086971283:
                                                if sample.get('rate', 0) <= 6.22306038167153e-06:
                                                    return 3
                                                else:
                                                    if sample.get('synack', 0) <= 0.03314627520740032:
                                                        return 5
                                                    else:
                                                        return 3
                                            else:
                                                return 3
                            else:
                                if sample.get('djit', 0) <= 0.006131461821496487:
                                    return 3
                                else:
                                    if sample.get('dinpkt', 0) <= 0.0015831506461836398:
                                        return 6
                                    else:
                                        if sample.get('synack', 0) <= 0.03183785267174244:
                                            if sample.get('sinpkt', 0) <= 0.0022897006710991263:
                                                if sample.get('sinpkt', 0) <= 0.0010980955266859382:
                                                    return 3
                                                else:
                                                    return 3
                                            else:
                                                return 3
                                        else:
                                            return 4
    else:
        if sample.get('sbytes', 0) <= 6.4782402660057414e-06:
            if sample.get('sbytes', 0) <= 6.19960633230221e-06:
                if sample.get('spkts', 0) <= 4.6970410039648414e-05:
                    return 2
                else:
                    return 6
            else:
                return 5
        else:
            if sample.get('ct_state_ttl', 0) <= 0.2500000074505806:
                if sample.get('sttl', 0) <= 0.6196078434586525:
                    if sample.get('response_body_len', 0) <= 9.870528811006807e-05:
                        return 6
                    else:
                        return 3
                else:
                    if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                        if sample.get('ct_srv_dst', 0) <= 0.04098360612988472:
                            if sample.get('stcpb', 0) <= 0.24292433261871338:
                                return 3
                            else:
                                return 6
                        else:
                            if sample.get('dpkts', 0) <= 0.0017244508489966393:
                                if sample.get('ct_srv_src', 0) <= 0.04032257944345474:
                                    if sample.get('dinpkt', 0) <= 0.07369474263396114:
                                        if sample.get('sinpkt', 0) <= 0.0023954269709065557:
                                            return 6
                                        else:
                                            return 3
                                    else:
                                        return 3
                                else:
                                    return 4
                            else:
                                return 3
                    else:
                        if sample.get('tcprtt', 0) <= 0.026146124117076397:
                            if sample.get('tcprtt', 0) <= 0.020411151461303234:
                                return 6
                            else:
                                return 4
                        else:
                            if sample.get('dbytes', 0) <= 4.4959822844248265e-05:
                                return 6
                            else:
                                return 3
            else:
                if sample.get('ct_srv_src', 0) <= 0.13709677010774612:
                    if sample.get('proto', 0) <= 0.3692307770252228:
                        if sample.get('sload', 0) <= 0.1837509348988533:
                            if sample.get('ct_srv_src', 0) <= 0.05645161122083664:
                                return 3
                            else:
                                if sample.get('proto', 0) <= 0.06923077069222927:
                                    return 4
                                else:
                                    if sample.get('spkts', 0) <= 0.00018788164015859365:
                                        if sample.get('ct_dst_ltm', 0) <= 0.06034482829272747:
                                            return 3
                                        else:
                                            if sample.get('ct_src_dport_ltm', 0) <= 0.06896551884710789:
                                                return 0
                                            else:
                                                if sample.get('proto', 0) <= 0.20000001043081284:
                                                    return 3
                                                else:
                                                    return 1
                                    else:
                                        return 3
                        else:
                            return 3
                    else:
                        if sample.get('sbytes', 0) <= 9.264580057788407e-06:
                            return 3
                        else:
                            if sample.get('rate', 0) <= 0.10555555298924446:
                                if sample.get('sloss', 0) <= 0.00018800525867845863:
                                    if sample.get('dmean', 0) <= 0.265666663646698:
                                        if sample.get('dur', 0) <= 1.1666668697785099e-07:
                                            return 4
                                        else:
                                            if sample.get('smean', 0) <= 0.22331080585718155:
                                                if sample.get('ct_dst_ltm', 0) <= 0.06034482829272747:
                                                    if sample.get('ct_src_ltm', 0) <= 0.12711863964796066:
                                                        return 3
                                                    else:
                                                        return 2
                                                else:
                                                    if sample.get('ct_srv_dst', 0) <= 0.1147540956735611:
                                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.10810810327529907:
                                                            return 3
                                                        else:
                                                            return 1
                                                    else:
                                                        return 3
                                            else:
                                                return 3
                                    else:
                                        return 3
                                else:
                                    return 3
                            else:
                                if sample.get('ct_src_ltm', 0) <= 0.31355932354927063:
                                    if sample.get('proto', 0) <= 0.4076923131942749:
                                        return 2
                                    else:
                                        if sample.get('proto', 0) <= 0.8961538374423981:
                                            if sample.get('rate', 0) <= 0.1547618955373764:
                                                if sample.get('proto', 0) <= 0.45000000298023224:
                                                    return 3
                                                else:
                                                    if sample.get('proto', 0) <= 0.7192307710647583:
                                                        if sample.get('sload', 0) <= 0.02033843006938696:
                                                            return 2
                                                        else:
                                                            return 3
                                                    else:
                                                        return 2
                                            else:
                                                return 3
                                        else:
                                            if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                                                if sample.get('ct_src_dport_ltm', 0) <= 0.043103449046611786:
                                                    if sample.get('rate', 0) <= 0.22500000149011612:
                                                        return 2
                                                    else:
                                                        if sample.get('ct_srv_dst', 0) <= 0.05737704783678055:
                                                            return 2
                                                        else:
                                                            return 3
                                                else:
                                                    return 3
                                            else:
                                                if sample.get('ct_src_ltm', 0) <= 0.17796610295772552:
                                                    return 2
                                                else:
                                                    if sample.get('ct_dst_ltm', 0) <= 0.06034482829272747:
                                                        if sample.get('sload', 0) <= 0.017927950248122215:
                                                            if sample.get('ct_dst_ltm', 0) <= 0.043103449046611786:
                                                                return 2
                                                            else:
                                                                return 3
                                                        else:
                                                            if sample.get('sload', 0) <= 0.03796506766229868:
                                                                if sample.get('rate', 0) <= 0.1339285671710968:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.19491524994373322:
                                                                        return 3
                                                                    else:
                                                                        return 2
                                                                else:
                                                                    return 3
                                                            else:
                                                                if sample.get('ct_dst_src_ltm', 0) <= 0.13709677010774612:
                                                                    return 2
                                                                else:
                                                                    return 3
                                                    else:
                                                        if sample.get('rate', 0) <= 0.1180555522441864:
                                                            return 2
                                                        else:
                                                            if sample.get('rate', 0) <= 0.2499999850988388:
                                                                if sample.get('sload', 0) <= 0.02214629016816616:
                                                                    if sample.get('ct_src_dport_ltm', 0) <= 0.06034482829272747:
                                                                        return 2
                                                                    else:
                                                                        return 3
                                                                else:
                                                                    return 2
                                                            else:
                                                                if sample.get('ct_src_dport_ltm', 0) <= 0.06034482829272747:
                                                                    return 3
                                                                else:
                                                                    return 2
                                else:
                                    if sample.get('sload', 0) <= 0.04049607366323471:
                                        if sample.get('ct_src_ltm', 0) <= 0.35593220591545105:
                                            if sample.get('sload', 0) <= 0.02214629016816616:
                                                if sample.get('ct_src_dport_ltm', 0) <= 0.043103449046611786:
                                                    return 2
                                                else:
                                                    if sample.get('rate', 0) <= 0.1180555522441864:
                                                        return 2
                                                    else:
                                                        return 3
                                            else:
                                                return 2
                                        else:
                                            return 2
                                    else:
                                        return 2
                else:
                    if sample.get('sbytes', 0) <= 3.420232314965688e-05:
                        if sample.get('ct_dst_ltm', 0) <= 0.1465517207980156:
                            if sample.get('sbytes', 0) <= 1.156331063612015e-05:
                                return 3
                            else:
                                if sample.get('proto', 0) <= 0.08076923340559006:
                                    if sample.get('sload', 0) <= 0.053151099011301994:
                                        if sample.get('ct_srv_src', 0) <= 0.2338709607720375:
                                            return 3
                                        else:
                                            if sample.get('sload', 0) <= 0.02151353983208537:
                                                if sample.get('ct_src_dport_ltm', 0) <= 0.06896551884710789:
                                                    return 3
                                                else:
                                                    return 4
                                            else:
                                                return 3
                                    else:
                                        return 3
                                else:
                                    if sample.get('rate', 0) <= 0.75:
                                        if sample.get('proto', 0) <= 0.1576923131942749:
                                            if sample.get('ct_srv_src', 0) <= 0.18548386543989182:
                                                return 3
                                            else:
                                                if sample.get('ct_dst_sport_ltm', 0) <= 0.12162161991000175:
                                                    return 3
                                                else:
                                                    if sample.get('proto', 0) <= 0.14230769872665405:
                                                        if sample.get('proto', 0) <= 0.11153846606612206:
                                                            return 0
                                                        else:
                                                            return 3
                                                    else:
                                                        return 3
                                        else:
                                            if sample.get('ct_dst_src_ltm', 0) <= 0.04032257944345474:
                                                return 3
                                            else:
                                                if sample.get('rate', 0) <= 0.08712121099233627:
                                                    if sample.get('ct_srv_dst', 0) <= 0.23770491033792496:
                                                        if sample.get('ct_src_ltm', 0) <= 0.10169491544365883:
                                                            if sample.get('sinpkt', 0) <= 3.916014534866008e-07:
                                                                return 3
                                                            else:
                                                                return 4
                                                        else:
                                                            return 3
                                                    else:
                                                        if sample.get('rate', 0) <= 0.041666874074138605:
                                                            return 0
                                                        else:
                                                            return 4
                                                else:
                                                    if sample.get('ct_src_ltm', 0) <= 0.16101694852113724:
                                                        if sample.get('proto', 0) <= 0.8730769157409668:
                                                            if sample.get('rate', 0) <= 0.1339285671710968:
                                                                if sample.get('proto', 0) <= 0.807692289352417:
                                                                    if sample.get('ct_src_ltm', 0) <= 0.0762711875140667:
                                                                        if sample.get('ct_srv_dst', 0) <= 0.2540983557701111:
                                                                            return 4
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.12711864337325096:
                                                                            if sample.get('ct_srv_src', 0) <= 0.3467741906642914:
                                                                                if sample.get('proto', 0) <= 0.5192307829856873:
                                                                                    if sample.get('ct_dst_ltm', 0) <= 0.06896551884710789:
                                                                                        if sample.get('ct_srv_src', 0) <= 0.2177419289946556:
                                                                                            if sample.get('dur', 0) <= 1.1666668697785099e-07:
                                                                                                return 3
                                                                                            else:
                                                                                                return 4
                                                                                        else:
                                                                                            return 4
                                                                                    else:
                                                                                        return 3
                                                                                else:
                                                                                    if sample.get('proto', 0) <= 0.7576923072338104:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('sload', 0) <= 0.014495754148811102:
                                                                                            return 3
                                                                                        else:
                                                                                            return 3
                                                                            else:
                                                                                if sample.get('rate', 0) <= 0.10101009905338287:
                                                                                    return 4
                                                                                else:
                                                                                    if sample.get('sload', 0) <= 0.017927950248122215:
                                                                                        if sample.get('proto', 0) <= 0.3500000089406967:
                                                                                            return 4
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        return 4
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('ct_srv_src', 0) <= 0.29032257199287415:
                                                                        if sample.get('proto', 0) <= 0.8423076868057251:
                                                                            return 3
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        return 4
                                                            else:
                                                                if sample.get('proto', 0) <= 0.41923077404499054:
                                                                    if sample.get('ct_srv_src', 0) <= 0.3145161271095276:
                                                                        if sample.get('ct_dst_src_ltm', 0) <= 0.08870967477560043:
                                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                                                                                if sample.get('ct_srv_dst', 0) <= 0.2049180343747139:
                                                                                    return 4
                                                                                else:
                                                                                    return 3
                                                                            else:
                                                                                if sample.get('proto', 0) <= 0.2884615510702133:
                                                                                    if sample.get('rate', 0) <= 0.238095223903656:
                                                                                        return 3
                                                                                    else:
                                                                                        return 4
                                                                                else:
                                                                                    if sample.get('rate', 0) <= 0.2916666567325592:
                                                                                        return 3
                                                                                    else:
                                                                                        return 0
                                                                        else:
                                                                            return 0
                                                                    else:
                                                                        if sample.get('dur', 0) <= 1.250000245534011e-07:
                                                                            return 0
                                                                        else:
                                                                            return 3
                                                                else:
                                                                    if sample.get('proto', 0) <= 0.6923076808452606:
                                                                        if sample.get('ct_srv_dst', 0) <= 0.2540983557701111:
                                                                            if sample.get('rate', 0) <= 0.22500000149011612:
                                                                                if sample.get('ct_srv_dst', 0) <= 0.2295081987977028:
                                                                                    if sample.get('ct_srv_dst', 0) <= 0.19672130793333054:
                                                                                        return 3
                                                                                    else:
                                                                                        return 0
                                                                                else:
                                                                                    return 4
                                                                            else:
                                                                                return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('ct_src_ltm', 0) <= 0.12711864337325096:
                                                                            if sample.get('sload', 0) <= 0.02784105110913515:
                                                                                return 4
                                                                            else:
                                                                                if sample.get('ct_srv_dst', 0) <= 0.24590163677930832:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.09322033822536469:
                                                                                        return 0
                                                                                    else:
                                                                                        return 3
                                                                        else:
                                                                            return 1
                                                        else:
                                                            if sample.get('ct_src_ltm', 0) <= 0.14406779408454895:
                                                                if sample.get('proto', 0) <= 0.9115384519100189:
                                                                    if sample.get('rate', 0) <= 0.10555555298924446:
                                                                        if sample.get('ct_dst_src_ltm', 0) <= 0.09677419066429138:
                                                                            if sample.get('proto', 0) <= 0.9000000059604645:
                                                                                if sample.get('ct_src_ltm', 0) <= 0.06779661029577255:
                                                                                    return 3
                                                                                else:
                                                                                    return 1
                                                                            else:
                                                                                if sample.get('ct_srv_src', 0) <= 0.15322580188512802:
                                                                                    if sample.get('sload', 0) <= 0.014495754148811102:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('ct_dst_ltm', 0) <= 0.06896551884710789:
                                                                                            return 3
                                                                                        else:
                                                                                            return 3
                                                                                else:
                                                                                    if sample.get('ct_srv_src', 0) <= 0.2338709607720375:
                                                                                        if sample.get('ct_src_dport_ltm', 0) <= 0.043103449046611786:
                                                                                            return 3
                                                                                        else:
                                                                                            if sample.get('ct_src_ltm', 0) <= 0.0762711875140667:
                                                                                                return 4
                                                                                            else:
                                                                                                return 3
                                                                                    else:
                                                                                        return 3
                                                                        else:
                                                                            return 3
                                                                    else:
                                                                        if sample.get('sload', 0) <= 0.04429258033633232:
                                                                            if sample.get('ct_srv_src', 0) <= 0.15322580188512802:
                                                                                if sample.get('sload', 0) <= 0.02033843006938696:
                                                                                    return 3
                                                                                else:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.09322033822536469:
                                                                                        if sample.get('rate', 0) <= 0.1964285671710968:
                                                                                            return 4
                                                                                        else:
                                                                                            return 3
                                                                                    else:
                                                                                        if sample.get('ct_dst_ltm', 0) <= 0.06896551884710789:
                                                                                            return 3
                                                                                        else:
                                                                                            return 4
                                                                            else:
                                                                                if sample.get('ct_dst_ltm', 0) <= 0.07758620753884315:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.09322033822536469:
                                                                                        if sample.get('ct_srv_src', 0) <= 0.16935483366250992:
                                                                                            if sample.get('rate', 0) <= 0.1339285671710968:
                                                                                                return 3
                                                                                            else:
                                                                                                return 4
                                                                                        else:
                                                                                            return 4
                                                                                    else:
                                                                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.06756756640970707:
                                                                                            return 4
                                                                                        else:
                                                                                            if sample.get('ct_srv_src', 0) <= 0.2016128972172737:
                                                                                                if sample.get('ct_dst_ltm', 0) <= 0.06034482829272747:
                                                                                                    return 4
                                                                                                else:
                                                                                                    return 3
                                                                                            else:
                                                                                                return 4
                                                                                else:
                                                                                    if sample.get('rate', 0) <= 0.1339285671710968:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('ct_srv_src', 0) <= 0.2177419289946556:
                                                                                            if sample.get('ct_dst_src_ltm', 0) <= 0.07258064299821854:
                                                                                                return 0
                                                                                            else:
                                                                                                return 3
                                                                                        else:
                                                                                            return 0
                                                                        else:
                                                                            if sample.get('ct_src_ltm', 0) <= 0.05932203307747841:
                                                                                return 3
                                                                            else:
                                                                                if sample.get('ct_src_ltm', 0) <= 0.11016948893666267:
                                                                                    return 4
                                                                                else:
                                                                                    if sample.get('ct_src_ltm', 0) <= 0.12711864337325096:
                                                                                        return 3
                                                                                    else:
                                                                                        if sample.get('ct_dst_ltm', 0) <= 0.06896551884710789:
                                                                                            return 3
                                                                                        else:
                                                                                            return 1
                                                                else:
                                                                    if sample.get('ct_dst_sport_ltm', 0) <= 0.10810810327529907:
                                                                        if sample.get('rate', 0) <= 0.09545454755425453:
                                                                            return 4
                                                                        else:
                                                                            if sample.get('proto', 0) <= 0.9576922953128815:
                                                                                if sample.get('proto', 0) <= 0.9307692050933838:
                                                                                    return 1
                                                                                else:
                                                                                    return 4
                                                                            else:
                                                                                if sample.get('ct_src_ltm', 0) <= 0.0762711875140667:
                                                                                    return 3
                                                                                else:
                                                                                    return 4
                                                                    else:
                                                                        if sample.get('ct_srv_dst', 0) <= 0.21311475336551666:
                                                                            return 3
                                                                        else:
                                                                            return 4
                                                            else:
                                                                return 4
                                                    else:
                                                        if sample.get('proto', 0) <= 0.7038461565971375:
                                                            return 4
                                                        else:
                                                            if sample.get('rate', 0) <= 0.09545454755425453:
                                                                return 4
                                                            else:
                                                                if sample.get('ct_srv_src', 0) <= 0.2338709607720375:
                                                                    if sample.get('rate', 0) <= 0.1339285671710968:
                                                                        if sample.get('sload', 0) <= 0.017927950248122215:
                                                                            return 3
                                                                        else:
                                                                            return 4
                                                                    else:
                                                                        if sample.get('ct_dst_sport_ltm', 0) <= 0.10810810327529907:
                                                                            return 3
                                                                        else:
                                                                            if sample.get('sload', 0) <= 0.02982969768345356:
                                                                                return 4
                                                                            else:
                                                                                return 3
                                                                else:
                                                                    return 1
                                    else:
                                        if sample.get('ct_srv_src', 0) <= 0.2822580561041832:
                                            return 4
                                        else:
                                            return 1
                        else:
                            if sample.get('ct_srv_dst', 0) <= 0.180327869951725:
                                if sample.get('proto', 0) <= 0.07307692430913448:
                                    return 3
                                else:
                                    return 2
                            else:
                                if sample.get('proto', 0) <= 0.14615384861826897:
                                    return 3
                                else:
                                    if sample.get('proto', 0) <= 0.6192307770252228:
                                        if sample.get('proto', 0) <= 0.257692314684391:
                                            return 3
                                        else:
                                            if sample.get('proto', 0) <= 0.45384617149829865:
                                                return 4
                                            else:
                                                return 3
                                    else:
                                        return 0
                    else:
                        return 6
