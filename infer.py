'''
Script to use ViTSTR to convert scene text image to text.

Usage:
    python3 infer.py --image demo_image/demo_1.png --model https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug_infer.pth

--image: path to image file to convert to text

Inference timing:
    Quantized on CPU:
        python3 infer.py --model vitstr_small_patch16_quant.pt --time --quantized
        Average inference time per image: 2.22e-02 sec

    CPU:
        python3 infer.py --model vitstr_small_patch16_224_aug_infer.pth --time 
        Average inference time per image: 3.24e-02 sec

        With JIT:
            python3 infer.py --model vitstr_small_patch16_jit.pt --time 
            Average inference time per image: 2.75e-02 sec

    GPU:
        python3 infer.py --model vitstr_small_patch16_224_aug_infer.pth --time --gpu
        Average inference time per image: 3.50e-03 sec
        
        With JIT:
            python3 infer.py --model vitstr_small_patch16_jit.pt --time --gpu
            Average inference time per image: 2.56e-03 sec

    RPi 4 CPU Quantized:
        python3 infer.py --model vitstr_small_patch16_quant.pt --time --rpi --quantized
        Average inference time per image: 3.59e-01 sec

    RPi 4 CPU JIT:
        python3 infer.py --model vitstr_small_patch16_jit.pt  --time --rpi
        Average inference time per image: 4.64e-01 sec
        

To generate torchscript jit
model.py
    def forward(self, input, seqlen: int =25): #text, is_train=True, seqlen=25):
        """ Transformation stage """
        #if not self.stages['Trans'] == "None":
        #    input = self.Transformation(input)

        #if self.stages['ViTSTR']:
        prediction = self.vitstr(input, seqlen=seqlen)
        return prediction


modules/vitstr.py
    def forward(self, x, seqlen: int =25):

'''

import os
import torch
import string
import validators
import time
from infer_utils import TokenLabelConverter, NormalizePAD,  ViTSTRFeatureExtractor
from infer_utils import get_args
from model import Model

def img2text(model, images, converter):
    pred_strs = []
    with torch.no_grad():
        for img in images:
            pred = model(img, seqlen=converter.batch_max_length)
            # print(pred.shape)#torch.Size([1, 27, 96])
            # print(pred)
            _, pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
            # print(pred_index.shape)#torch.Size([1, 27, 1])
            # print(pred_index)
            pred_index = pred_index.view(-1, converter.batch_max_length)# view(-1, converter.batch_max_length) : [b, 25] -> [b, 25]
            # print(pred_index.shape)#torch.Size([1, 27])
            # print(pred_index)
            length_for_pred = torch.IntTensor([converter.batch_max_length - 1] )
            # print(length_for_pred.shape)#torch.Size([1])
            # print(length_for_pred)
            # print(pred_index[:,1:])
            # print(pred_index[:,1:].shape)#torch.Size([1, 26])
            pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
            # print(pred_str)
            pred_EOS = pred_str[0].find('[s]')# find : return index
            pred_str = pred_str[0][:pred_EOS]
            # print(pred_str)

            pred_strs.append(pred_str)

    return pred_strs

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    converter = TokenLabelConverter(args)
    args.num_class = len(converter.character)
    print('Number of classes:', args.num_class)
    extractor = ViTSTRFeatureExtractor()
    if args.time:
        files = ["demo_1.png", "demo_2.jpg", "demo_3.png",  "demo_4.png",  "demo_5.png",  "demo_6.png",  "demo_7.png",  "demo_8.jpg", "demo_9.jpg", "demo_10.jpg"]
        images = []
        extractor
        for f in files:
            f = os.path.join("demo_image", f)
            img = extractor(f)
            if args.gpu:
                img = img.to(device)
            images.append(img)
    else:
        assert(args.image is not None)
        files = [args.image]
        img = extractor(args.image)
        if args.gpu:
            img = img.to(device)
        images = [img]

    if args.quantized:
        if args.rpi:
            backend = "qnnpack"   #arm
        else:
            backend =  "fbgemm"   #x86

        torch.backends.quantized.engine = backend
    
    if validators.url(args.model):
        checkpoint = args.model.rsplit('/', 1)[-1] # rsplit('/', 1)[]
        torch.hub.download_url_to_file(args.model, checkpoint)
    else:
        checkpoint = args.model

    if args.quantized:
        model = torch.jit.load(checkpoint)
    else:
        model = Model(args)
        model = torch.nn.DataParallel(model).to(device)

    model.load_state_dict(torch.load(args.saved_model, map_location=device))

    model.eval()

    if args.time:
        n_times = 10
        n_total = len(images) * n_times
        [img2text(model, images, converter) for _ in range(n_times)]
        start_time = time.time()
        [img2text(model, images, converter) for _ in range(n_times)]
        end_time = time.time()
        ave_time = (end_time - start_time) / n_total
        print("Average inference time per image: %0.2e sec" % ave_time) 

    pred_strs = img2text(model, images, converter)

    return zip(files, pred_strs)


if __name__ == '__main__':
    args = get_args()
    args.character = '0123456789abcdefghijklmnopqrstuvwxyz가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘?!' # string.printable[:-6] # removes whitespace from string.printable
    print(args.character)
    data = infer(args)
    for filename, text in data:
        print(filename, "\t: ", text)

    #print(infer(args))
