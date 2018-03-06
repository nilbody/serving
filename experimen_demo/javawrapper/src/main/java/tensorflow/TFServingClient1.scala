package tensorflow

import java.io.{BufferedReader, InputStream, InputStreamReader}
import java.lang.{Double, Long}
import java.text.SimpleDateFormat
import java.util
import java.util.Calendar
import java.util.concurrent.{ExecutorService, Executors}
import java.util.regex.Pattern

import com.google.protobuf.ByteString
import com.hankcs.hanlp.HanLP
import com.hankcs.hanlp.dictionary.CustomDictionary
import com.hankcs.hanlp.dictionary.py.Pinyin
import com.hankcs.hanlp.tokenizer.NLPTokenizer
import io.grpc.{ManagedChannel, ManagedChannelBuilder}
import org.tensorflow.example._
import org.tensorflow.framework.{DataType, TensorProto, TensorShapeProto}
import tensorflow.serving.Model.ModelSpec
import tensorflow.serving.Predict.PredictRequest
import tensorflow.serving.PredictionServiceGrpc._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source._

/**
  * 增加杰卡德距离
  */

object TFServingClient1{

  private final var channel : ManagedChannel = _
  private final var stub : PredictionServiceBlockingStub = _
  private final var host = "127.0.0.1"
  private final var port = 8500
  private final var model_name = "wide_and_deep"
  private final var signature_name = "test_model"

  private def createByteStringTensorProto(values: Seq[ByteString]): TensorProto = {
    val dim = TensorShapeProto.Dim.newBuilder()
      .setSize(values.size)

    val shape = TensorShapeProto.newBuilder()
      .addDim(dim)

    val builder = TensorProto.newBuilder()
      .setDtype(DataType.DT_STRING)
      .setTensorShape(shape)

    values.foreach( value =>
      builder.addStringVal(value)
    )

    builder.build()
  }

  private def createFloatFeature(values: Seq[Float]): Feature = {
    val feature_builder = Feature.newBuilder()
    val list_builder = org.tensorflow.example.FloatList.newBuilder()
    values.foreach(t => list_builder.addValue(t))
    feature_builder.setFloatList(list_builder).build()
  }

  private def createIntFeature(values: Seq[Int]): Feature = {
    val feature_builder = Feature.newBuilder()
    val list_builder = org.tensorflow.example.Int64List.newBuilder()
    values.foreach(t => list_builder.addValue(t))
    feature_builder.setInt64List(list_builder).build()
  }

  private def createByteStringFeature(values: Seq[ByteString]): Feature = {
    val feature_builder = Feature.newBuilder()
    val list_builder = org.tensorflow.example.BytesList.newBuilder()
    values.foreach(t => list_builder.addValue(t))
    feature_builder.setBytesList(list_builder).build()
  }

  private def createRequest(input_strings : Seq[ByteString]): PredictRequest= {
    val modelSpec = ModelSpec.newBuilder()
      .setName(model_name)

    val requestBuilder = PredictRequest.newBuilder()
      .setModelSpec(modelSpec)
      .putInputs("inputs", createByteStringTensorProto(input_strings))

    requestBuilder.build()
  }

  private def getProbabilities(request: PredictRequest): Seq[Float] = {

    if(channel.isShutdown || channel.isTerminated){
      this.synchronized{
        if(channel.isShutdown || channel.isTerminated){
          channel = ManagedChannelBuilder
            .forAddress(host, port)
            .usePlaintext(true)
            .asInstanceOf[ManagedChannelBuilder[_]]
            .build()
        }
      }
    }

    newBlockingStub(channel).predict(request)
      .getOutputsOrThrow("scores")
      .getFloatValList
      .asScala
      .map(_.toFloat)
      .zipWithIndex
      .collect{
        case t if t._2 % 2 == 1 => t._1
      }
  }


  private def toWeek(date: String): String = {
    val formatter_day: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd")
    val d = formatter_day.parse(date)
    val calendar = Calendar.getInstance()
    calendar.setTime(d)
    val week = calendar.get(Calendar.DAY_OF_WEEK) -1
    week.toString
  }

  private def addWords{
    val fis: InputStream = this.getClass.getClassLoader.getResourceAsStream("ext.dic")
    val br = new BufferedReader(new InputStreamReader(fis))
    //先读一次，看是否为空
    var str:String = br.readLine()
    while(str != null){
      CustomDictionary.add(str.trim)
      str = br.readLine()
    }
    br.close()
  }

  val regex = "[a-zA-Z0-9 \u4e00-\u9fa5]+"
  val pattern: Pattern = Pattern.compile(regex)
  private def haNlpfenci(words: String,isTopin:Boolean) = {
    var str = ""

    val matcher = pattern.matcher(HanLP.convertToSimplifiedChinese(words.toLowerCase.replaceAll("[\\pP]"," ")))
    while (matcher.find()) {
      str += matcher.group(0)
    }
    val titleTermList = NLPTokenizer.segment(str)
    //得到title分词，并过滤掉停用词及标点符号
    val titleWords = (for(i <- 0 until  titleTermList.size) yield {titleTermList.get(i).word.replaceAll(" ","")}).filter(t=>t != "" && t != " " && !(t.matches("\\d+") && t.length >5) ).take(30)
    if(isTopin){
      val appdanzi = titleWords.map(t=>{if(t.matches("[\u4e00-\u9fa5]+")) {t.mkString(";")}else{t}}).mkString(";").split(";").take(4).mkString(";")
      val pingyin = HanLP.convertToPinyinString(appdanzi,"",false)
      Array(appdanzi,titleWords.filter(t=> !appdanzi.contains(t)).mkString(";"),pingyin).filter(t=> !t.equals("")).mkString(";")
    }else{
      val appdanzi = titleWords.map(t=>{if(t.matches("[\u4e00-\u9fa5]+")) {t.mkString(";")}else{t}}).mkString(";").split(";").take(4).mkString(";")
      Array(appdanzi,titleWords.filter(t=> !appdanzi.contains(t)).mkString(";")).filter(t=> !t.equals("")).mkString(";")
    }
  }

  def startClient(): Unit = {
    addWords
    if(channel != null){
      channel.shutdownNow()
    }
    
    channel = ManagedChannelBuilder
      .forAddress(host, port)
      .usePlaintext(true)
      .asInstanceOf[ManagedChannelBuilder[_]]
      .build()
}

  def stopClient() = {
    if(channel != null){
      channel.shutdownNow()
    }
  }

  /**
    * 将小时编码，为了和week这个字段一起使用，小时00编码成8，01编码成9，依此类推
    *
    * @param hour
    * @return
    */
  private def encodeHour(hour:String):String = {
    val encoded_hour = (hour.toInt + 8).toString
    encoded_hour
  }

  /**
    * 对下载人数进行onr-hot，一共6位，取对数之后[0,2）的第一位是1，>=6的第6位是1
    * @param download_count:下载人数，取对数之后的值
    * @return
    */
  def downloadOneHotV2(download_count:Float) = {
    val zerolist = Array(0,0,0,0,0,0)
    val transformed_download_count = math.floor(download_count)
    if (transformed_download_count >= 6) {
      zerolist(5) = 1
      //println("value of position 5 is " + zerolist(5) )
    }
    else {
      if (transformed_download_count < 2){
        zerolist(0) = 1
        //println("value of position 0 is " + zerolist(0) )
      }
      else{
        for (idx <- 2 to 6) {
          if (idx == transformed_download_count){
            zerolist(idx-1) = 1
            //println("value of position " + idx + " is " + zerolist(idx-1) )
          }
        }
      }
    }
    //println("every one of one hot is:")
    //zerolist.foreach(println)
    //println("---------------one hot download is: " + zerolist.mkString(";"))
    zerolist.mkString(";")
  }

  private def getSingleCharacters(inputStr:String)={
    val regex = "[a-zA-Z0-9 \u4e00-\u9fa5]+"
    var temp = ""

    val m = Pattern.compile(regex).matcher(HanLP.convertToSimplifiedChinese(inputStr.toLowerCase.replaceAll("[\\pP]"," ")))
    while (m.find()) {
      temp += m.group(0)
    }
    val titleTermList = NLPTokenizer.segment(temp)
    val titleWords = (for(i <- 0 until  titleTermList.size) yield {titleTermList.get(i).word.replaceAll(" ","")}).filter(t=>t != "" && t != " " && !(t.matches("\\d+") && t.size>5))
    val wordlist = titleWords.map(t=>{if(t.matches("[\u4e00-\u9fa5]+")) {t.mkString(";")}else{t}}).mkString(";").split(";").toList
    wordlist
  }

  /**
    *Input: a list of words, e.g., ['I', 'am', 'Denny']
    * Output: a list of unigram
    *
    * @param words
    */
  private def getUniGram(words:List[String])={
    assert (words.isInstanceOf[List[String]])
    words
  }

  /**
    * Input: a list of words, e.g., ['I', 'am', 'Denny']
    * Output: a list of bigram, e.g., ['I_am', 'am_Denny']
    * I use _ as join_string for this example.
    *
    * @param words:a list of words
    */
  private def getBiGram(words:List[String], join_string:String="")={
    val word_num = words.size
    if(word_num > 1){
      words.sliding(2).toList.map(ele=>ele.mkString(join_string))
    }
    else getUniGram(words)
  }

  /**
    * 计算两个字符传数组的jaccard距离
    *
    * @param A
    * @param B
    * @return
    */
  private def JaccardCoef(A:List[String], B:List[String]):Double={
    val setA = A.toSet
    val setB = B.toSet
    setA.intersect(setB).size/setA.union(setB).size.toDouble
    //dist
  }

  /**
    * 计算Dice distance
    *
    * @param A
    * @param B
    * @return
    */
  private def DiceDist(A:List[String], B:List[String]):Double={
    val setA = A.toSet
    val setB = B.toSet
    2.0*setA.intersect(setB).size/(setA.size+setB.size.toFloat)
  }

  /**
    * 计算jaccard 或者Dice distance
    *
    * @param A
    * @param B
    * @param dist_mode
    * @return
    */
  private def computeDist(A:List[String], B:List[String],dist_mode:String="jaccard_coef"):Double={
    var dist = 0.0
    if(dist_mode == "jaccard_coef"){
      dist = JaccardCoef(A, B)
    }
    else if (dist_mode == "dice_dist") {
      dist = DiceDist(A,B)
    }
    dist
  }

  /** 或得字符串数组的每一个元素的拼音
    * e.g.
    * input: List(acfun,奇，虎，360)
    * output:List(acfun,qi, hu, 360)
    *
    * @param wordlist
    */
  private def getPinYin(wordlist:List[String]):List[String]={
    wordlist.map(ele=>HanLP.convertToPinyinString(ele,"",false))
  }

  /** 获取字符串中汉字的声母，其他字符扔掉
    *  * e.g.
    * input: List(acfun,奇，虎，360)
    * output:List(q,h)
    *
    * @param wordlist
    * @return
    */
  private def getShengmu(wordlist:List[String])={
    val pinyinlist: util.List[Pinyin] = HanLP.convertToPinyinList(wordlist.mkString(";"))
    var arr:List[String] = List()

    var p =0
    while (p < pinyinlist.size()){
      if (pinyinlist.get(p).getShengmu().toString != "none") { arr = arr :+ pinyinlist.get(p).getShengmu().toString}
      p+=1
    }
    arr
  }

  /** 求两个整数的最小值
    *
    * @param x
    * @param y
    * @return
    */
  private def twoEleMinum(x:Int,y:Int) = {
    var res = 0
    if (x < y){res = x}
    else{res = y}
    res
  }
  /** 求三个个整数的最小值
    *
    * @param x
    * @param y
    * @param z
    * @return
    */
  private  def minimum(x: Int, y: Int, z: Int) = twoEleMinum(twoEleMinum(x,y), z)

  /** 求两个字符串的最小编辑距离
    *
    * @param str1
    * @param str2
    * @return
    */
  private def editDistance(str1: String, str2: String): Int = {
    val dist = Array.tabulate(str2.length + 1, str1.length + 1) { (i, j) => if (j == 0) i else if (i == 0) j else 0 }
    for (j <- 1 to str2.length; i <- 1 to str1.length)
      dist(j)(i) = if (str2(j - 1) == str1(i - 1)) dist(j - 1)(i - 1)
      else minimum(dist(j - 1)(i) + 1, dist(j)(i - 1) + 1, dist(j - 1)(i - 1) + 1)
    dist(str2.length)(str1.length)
  }

  /**
    *  相比computeScores1,去掉了拼音的编辑距离，同时week和hour进行交叉用168维度的one-hot(在Tensorflow侧实现)
    *  app的下载人数这个特征也是取对数后进行了one-hot，一共是6位
    *  用于本地测试模型
    * @param query
    * @param imeifea:imei
    * @param userFeature
    * @param appFeature
    * @return
    */
  def computeScores_v3(query:String, imeifea:String, userFeature:java.util.Map[String,String],
                       appFeature:util.ArrayList[Array[Feature]]): util.Map[java.lang.Long, java.lang.Double] = {

    //println("###############userFeature is: "+ userFeature)
    val queryword = haNlpfenci(query,false)//搜索词,进行分词
    val querywd = queryword.split(";") //将分词结果切割成数组
    //println("###############query is : "+ querywd.toList)

    val query_single_words = getSingleCharacters(query)
    val query_unigram = query_single_words
    val query_bigram = getBiGram(query_single_words)
    val query_pinyin = getPinYin(query_single_words)
    //val query_shengmu = getShengmu(query_single_words)

    val yyyyMMddFormater: SimpleDateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd HH")
    //样本数
    val date = yyyyMMddFormater.format(java.util.Calendar.getInstance.getTime)
    //val week = createIntFeature(if(!date.equals("") && date !=null) toWeek(date).split(";").map(_.toInt) else Array[Int](0))//只使用week特征的情况
    val week_and_hour = toWeek(date) + date.substring(11,13)//把week 和hour两个特征拼接在一起

    val week =  createByteStringFeature(week_and_hour.split(";").map(t=> ByteString.copyFrom(t.getBytes)))
    //println("week feature is:"+week)
    val query_indices = createByteStringFeature(querywd.map(t=> ByteString.copyFrom(t.getBytes)))
    val install_tags_indices = createIntFeature(if(userFeature.get("install") !=null && !userFeature.get("install").equals("")) userFeature.get("install").split(";").map(_.toInt) else Array[Int](0))
    val uninstall_tags_indices = createIntFeature(if(userFeature.get("uninstall") !=null && !userFeature.get("uninstall").equals("")) userFeature.get("uninstall").split(";").map(_.toInt) else Array[Int](0))
    val phonetype = createIntFeature(if(userFeature.get("phonetype") !=null && !userFeature.get("phonetype").equals("")) userFeature.get("phonetype").split(";").map(_.toInt) else Array[Int](0))
    val search_indices = createIntFeature(if(userFeature.get("search") !=null && !userFeature.get("search").equals("")) userFeature.get("search").split(";").map(_.toInt) else Array[Int](0))
    val appstore_search_indices = createIntFeature(if(userFeature.get("appstore_search") !=null && !userFeature.get("appstore_search").equals("")) userFeature.get("appstore_search").split(";").map(_.toInt) else Array[Int](0))
    val citys = createIntFeature(if(userFeature.get("citys") !=null && !userFeature.get("citys").equals("")) userFeature.get("citys").split(";").map(_.toInt) else Array[Int](0))
    val appuse_length = createIntFeature(if(userFeature.get("appuse_length") !=null && !userFeature.get("appuse_length").equals("")) userFeature.get("appuse_length").split(";").map(_.toInt) else Array[Int](0))
    val samples = mutable.ArrayBuilder.make[ByteString]

    val size = appFeature.size()
    val appids = mutable.ArrayBuilder.make[Long]

    var i = 0

    while (i < size) {
      val appid = appFeature.get(i)(0)
      val appNames0 = appFeature.get(i)(1)
      //println("####app appNames0:##############"+appNames0)
      val appfeature = appFeature.get(i)(2)
      //println("####app feature:##############"+appfeature)
      val apps_user_download_count = appFeature.get(i)(3).getFloatList.getValue(0)
      //println("####download original value from redis:"+apps_user_download_count)
      val appNames = appNames0.getBytesList.getValue(0).toStringUtf8
      val appfenci = haNlpfenci(appNames,true)
      //println(appfenci)
      val appname_indices = createByteStringFeature(appfenci.split(";").map(t => ByteString.copyFrom(t.getBytes)))
      val appname_single_words = getSingleCharacters(appNames)
      val appname_unigram = getUniGram(appname_single_words)
      val appname_bigram = getBiGram(appname_single_words)
      val appname_pinyin = getPinYin( appname_single_words)
      //val appname_shengmu = getShengmu(appname_single_words)

      val dist = Array(computeDist(query_unigram,appname_unigram).toDouble,computeDist(query_unigram,appname_unigram,dist_mode = "dice_dist").toDouble,
        computeDist(query_bigram,appname_bigram).toDouble,computeDist(query_bigram,appname_bigram,dist_mode = "dice_dist").toDouble,
        computeDist(query_pinyin,appname_pinyin).toDouble
      )
      //打印特征只能打印使用createXXXFeature转化之前的特征状态，否则是一串符号看不懂
      //println("----------------"+appid.getInt64List+"----------------------")
     // println("---------------- every dist of lsi dist is as follows:----------------------")
      //dist.foreach(println)
      val lsi = createFloatFeature(dist.map(_.toFloat))
      val download_count_onehot = downloadOneHotV2(apps_user_download_count.toFloat)
      //println("******************************************************")
      //println("one hot siz: "+ download_count_onehot.split(";").length+",value: " + download_count_onehot)
      val apps_user_download_count_onehot = createFloatFeature(download_count_onehot.split(";").map(_.toFloat))

      val input_feature_map = new util.HashMap[String, Feature]()
      input_feature_map.put("query_indices", query_indices)
      input_feature_map.put("appid", appid)
      input_feature_map.put("appname_indices", appname_indices)
      input_feature_map.put("appfeature", appfeature)
      input_feature_map.put("week", week)
      input_feature_map.put("lsi", lsi)
      input_feature_map.put("app_download_count",apps_user_download_count_onehot)
      input_feature_map.put("install_tags_indices", install_tags_indices)
      input_feature_map.put("uninstall_tags_indices", uninstall_tags_indices)
      input_feature_map.put("phonetype", phonetype)
      input_feature_map.put("search_indices", search_indices)
      input_feature_map.put("appstore_search_indices", appstore_search_indices)
      input_feature_map.put("citys", citys)
      input_feature_map.put("appuse_length", appuse_length)

      val features = Features.newBuilder().putAllFeature(input_feature_map).build()
      samples += Example.newBuilder().setFeatures(features).build().toByteString
      appids += appid.getInt64List.getValue(0)
      i += 1
    }
    val request = createRequest(samples.result())
    //println("pagesize"+request.getSerializedSize)
    val scores: Seq[Float] = getProbabilities(request)
    appids.result().zip(scores).sortBy(-_._2).take(100).map(t=>(java.lang.Long.valueOf(t._1),java.lang.Double.valueOf(t._2))).toMap.asJava
  }


  /**
    *
    * @param query
    * @param imeifea
    * @param userFeature
    * @param appFeature
    * @return
    */
  def computeScores_v2(query:String, imeifea:String, userFeature:java.util.Map[String,String],
                    appFeature:util.ArrayList[Array[Feature]]): util.Map[java.lang.Long, java.lang.Double] = {

    val queryword = haNlpfenci(query,false)//搜索词,进行分词
    val querywd = queryword.split(";") //将分词结果切割成数组

    println(querywd.toList)

    val query_single_words = getSingleCharacters(query)
    val query_unigram = query_single_words
    val query_bigram = getBiGram(query_single_words)
    val query_pinyin = getPinYin(query_single_words)
    val query_shengmu = getShengmu(query_single_words)

    val yyyyMMddFormater: SimpleDateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd HH")
    //样本数
    val date = yyyyMMddFormater.format(java.util.Calendar.getInstance.getTime)
    //val week = createIntFeature(if(!date.equals("") && date !=null) toWeek(date).split(";").map(_.toInt) else Array[Int](0))//只使用week特征的情况
    val week_and_hour = toWeek(date)+";" + encodeHour(date.substring(11,13))//把week 和hour两个特征拼接在一起
    val week = createIntFeature(if(!date.equals("") && date !=null) week_and_hour.split(";").map(_.toInt) else Array[Int](0))
    val query_indices = createByteStringFeature(querywd.map(t=> ByteString.copyFrom(t.getBytes)))
    val install_tags_indices = createIntFeature(if(userFeature.get("install") !=null && !userFeature.get("install").equals("")) userFeature.get("install").split(";").map(_.toInt) else Array[Int](0))
    val uninstall_tags_indices = createIntFeature(if(userFeature.get("uninstall") !=null && !userFeature.get("uninstall").equals("")) userFeature.get("uninstall").split(";").map(_.toInt) else Array[Int](0))
    val phonetype = createIntFeature(if(userFeature.get("phonetype") !=null && !userFeature.get("phonetype").equals("")) userFeature.get("phonetype").split(";").map(_.toInt) else Array[Int](0))
    val search_indices = createIntFeature(if(userFeature.get("search") !=null && !userFeature.get("search").equals("")) userFeature.get("search").split(";").map(_.toInt) else Array[Int](0))
    val appstore_search_indices = createIntFeature(if(userFeature.get("appstore_search") !=null && !userFeature.get("appstore_search").equals("")) userFeature.get("appstore_search").split(";").map(_.toInt) else Array[Int](0))
    val citys = createIntFeature(if(userFeature.get("citys") !=null && !userFeature.get("citys").equals("")) userFeature.get("citys").split(";").map(_.toInt) else Array[Int](0))
    val appuse_length = createIntFeature(if(userFeature.get("appuse_length") !=null && !userFeature.get("appuse_length").equals("")) userFeature.get("appuse_length").split(";").map(_.toInt) else Array[Int](0))
    val samples = mutable.ArrayBuilder.make[ByteString]

    val size = appFeature.size()
    val appids = mutable.ArrayBuilder.make[Long]

    var i = 0

    while (i < size) {
      val appid = appFeature.get(i)(0)
      val appNames0 = appFeature.get(i)(1)
      val appfeature = appFeature.get(i)(2)
      val apps_user_download_count = appFeature.get(i)(3).getFloatList.getValue(0)

      val appNames = appNames0.getBytesList.getValue(0).toStringUtf8
      val appfenci = haNlpfenci(appNames,true)
      //println(appfenci)
      val appname_indices = createByteStringFeature(appfenci.split(";").map(t => ByteString.copyFrom(t.getBytes)))

      val appname_single_words = getSingleCharacters(appNames)
      val appname_unigram = getUniGram(appname_single_words)
      val appname_bigram = getBiGram(appname_single_words)
      val appname_pinyin = getPinYin( appname_single_words)
      val appname_shengmu = getShengmu(appname_single_words)

      val dist = Array(computeDist(query_unigram,appname_unigram).toDouble,computeDist(query_unigram,appname_unigram,dist_mode = "dice_dist").toDouble,
        computeDist(query_bigram,appname_bigram).toDouble,computeDist(query_bigram,appname_bigram,dist_mode = "dice_dist").toDouble,
        computeDist(query_pinyin,appname_pinyin).toDouble,
        editDistance(query_pinyin.mkString,appname_pinyin.mkString).toDouble,editDistance(query_shengmu.mkString,appname_shengmu.mkString).toDouble,
        apps_user_download_count.toDouble
      )
      //println("----------------"+appid.getInt64List+"----------------------")
      //dist.foreach(println)
      val lsi = createFloatFeature(dist.map(_.toFloat))
      val input_feature_map = new util.HashMap[String, Feature]()
      input_feature_map.put("query_indices", query_indices)
      input_feature_map.put("appid", appid)
      input_feature_map.put("appname_indices", appname_indices)
      input_feature_map.put("appfeature", appfeature)
      input_feature_map.put("week", week)
      input_feature_map.put("lsi", lsi)
      input_feature_map.put("install_tags_indices", install_tags_indices)
      input_feature_map.put("uninstall_tags_indices", uninstall_tags_indices)
      input_feature_map.put("phonetype", phonetype)
      input_feature_map.put("search_indices", search_indices)
      input_feature_map.put("appstore_search_indices", appstore_search_indices)
      input_feature_map.put("citys", citys)
      input_feature_map.put("appuse_length", appuse_length)
      val features = Features.newBuilder().putAllFeature(input_feature_map).build()
      samples += Example.newBuilder().setFeatures(features).build().toByteString
      appids += appid.getInt64List.getValue(0)
      i += 1
    }

    val request = createRequest(samples.result())
    //println("pagesize"+request.getSerializedSize)
    val scores: Seq[Float] = getProbabilities(request)
    appids.result().zip(scores).sortBy(-_._2).take(100).map(t=>(java.lang.Long.valueOf(t._1),java.lang.Double.valueOf(t._2))).toMap.asJava
  }

  /**
    *
    * @param query 搜索词 type String
    * @param imeifea imei type String
    * @param userFeature 用户特征 type java.util.Map<String,String>
    * @param appFeature app特征 type Array<Array[Feature]>
    * @return
    */
  def computeScores_v1(query:String, imeifea:String, userFeature:java.util.Map[String,String],
                       appFeature:util.ArrayList[Array[Feature]]): util.Map[java.lang.Long, java.lang.Double] = {

    val querywd = haNlpfenci(query,false).split(";") //搜索词

    val firstQuery = querywd.take(1).mkString("")
    val sencondQuery = querywd.take(2).mkString("")

    val yyyyMMddFormater: SimpleDateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd")
    //样本数
    val date = yyyyMMddFormater.format(java.util.Calendar.getInstance.getTime)
    val week = createIntFeature(if(!date.equals("") && date !=null) toWeek(date).split(";").map(_.toInt) else Array[Int](0))
    val query_indices = createByteStringFeature(querywd.map(t=> ByteString.copyFrom(t.getBytes)))
    val install_tags_indices = createIntFeature(if(userFeature.get("install") !=null && !userFeature.get("install").equals("")) userFeature.get("install").split(";").map(_.toInt) else Array[Int](0))
    val uninstall_tags_indices = createIntFeature(if(userFeature.get("uninstall") !=null && !userFeature.get("uninstall").equals("")) userFeature.get("uninstall").split(";").map(_.toInt) else Array[Int](0))
    val phonetype = createIntFeature(if(userFeature.get("phonetype") !=null && !userFeature.get("phonetype").equals("")) userFeature.get("phonetype").split(";").map(_.toInt) else Array[Int](0))
    val search_indices = createIntFeature(if(userFeature.get("search") !=null && !userFeature.get("search").equals("")) userFeature.get("search").split(";").map(_.toInt) else Array[Int](0))
    val appstore_search_indices = createIntFeature(if(userFeature.get("appstore_search") !=null && !userFeature.get("appstore_search").equals("")) userFeature.get("appstore_search").split(";").map(_.toInt) else Array[Int](0))
    val citys = createIntFeature(if(userFeature.get("citys") !=null && !userFeature.get("citys").equals("")) userFeature.get("citys").split(";").map(_.toInt) else Array[Int](0))
    val appuse_length = createIntFeature(if(userFeature.get("appuse_length") !=null && !userFeature.get("appuse_length").equals("")) userFeature.get("appuse_length").split(";").map(_.toInt) else Array[Int](0))
    val samples = mutable.ArrayBuilder.make[ByteString]

    val size = appFeature.size()
    val appids = mutable.ArrayBuilder.make[Long]

    var i = 0

    while (i < size) {
      val appid = appFeature.get(i)(0)
      val appname_indices = appFeature.get(i)(1)
      val appfeature = appFeature.get(i)(2)
      //提取app名字
      val appname0: util.List[ByteString] = appname_indices.getBytesList.getValueList //明天来改啊
      val appname1 = mutable.ArrayBuilder.make[String]
      val appnamesize = appname0.size()

      var y = 0
      while (y < appnamesize) {
        appname1 += appname0.get(y).toStringUtf8
        y += 1
      }
      val appname = appname1.result()

      val firstAppname = appname.take(1).mkString("")
      val sencondAppname = appname.take(2).mkString("")

      val fourAppname = appname.take(4)

      var tmp = ""
      val fourPinyin = for (i <- fourAppname) yield {
        tmp = tmp + i; HanLP.convertToPinyinString(tmp, "", false) + ";" + HanLP.convertToPinyinFirstCharString(tmp, "", false)
      }

      val sim = if (sencondQuery.equals(sencondAppname)) {
        1
      } else if (HanLP.convertToPinyinString(sencondQuery, "", false).equals(HanLP.convertToPinyinString(sencondAppname, "", false))) {
        0.75
      } else if (firstQuery.equals(firstAppname)) {
        0.5
      } else if (HanLP.convertToPinyinString(firstQuery, "", false).equals(HanLP.convertToPinyinString(firstAppname, "", false)) || fourPinyin.mkString(";").contains(firstQuery)) {
        0.25
      } else {
        0.0
      }

      val querysize = querywd.length
      val querySet = querywd.toSet

      var cnt = 0
      var n = 0
      while (n < appnamesize) {
        if (querySet.contains(appname(n))) {
          cnt += 1
        }
        n += 1
      }

      val jacard = Array((cnt / ((querySet ++ appname.toSet).size + 0.01)).toFloat, sim.toFloat)

      val lsi = createFloatFeature(jacard)

      val input_feature_map = new util.HashMap[String, Feature]()
      input_feature_map.put("query_indices", query_indices)
      input_feature_map.put("appid", appid)
      input_feature_map.put("appname_indices", appname_indices)
      input_feature_map.put("appfeature", appfeature)
      input_feature_map.put("week", week)
      input_feature_map.put("lsi", lsi)
      input_feature_map.put("install_tags_indices", install_tags_indices)
      input_feature_map.put("uninstall_tags_indices", uninstall_tags_indices)
      input_feature_map.put("phonetype", phonetype)
      input_feature_map.put("search_indices", search_indices)
      input_feature_map.put("appstore_search_indices", appstore_search_indices)
      input_feature_map.put("citys", citys)
      input_feature_map.put("appuse_length", appuse_length)
      val features = Features.newBuilder().putAllFeature(input_feature_map).build()
      samples += Example.newBuilder().setFeatures(features).build().toByteString
      appids += appid.getInt64List.getValue(0)
      i += 1
    }

    val request = createRequest(samples.result())
    val scores: Seq[Float] = getProbabilities(request)
    appids.result().zip(scores).sortBy(-_._2).take(100).map(t=>(java.lang.Long.valueOf(t._1),java.lang.Double.valueOf(t._2))).toMap.asJava
  }

  //由这个入口来控制逻辑
  def computeScores(query:String, imeifea:String, userFeature:java.util.Map[String,String],
                     appFeature:util.ArrayList[Array[Feature]]): util.Map[java.lang.Long, java.lang.Double] = {

    var result:util.Map[java.lang.Long, java.lang.Double] = new util.HashMap[java.lang.Long, java.lang.Double]();
    if(appFeature.size() >0 && appFeature.get(0).size == 3){
      //v1,v2版本
      result = computeScores_v1(query:String, imeifea:String, userFeature:java.util.Map[String,String],
        appFeature:util.ArrayList[Array[Feature]])
    }else if(appFeature.size() >0 && appFeature.get(0).size == 4){
      //v1,v3版本
      result = computeScores_v3(query:String, imeifea:String, userFeature:java.util.Map[String,String],
        appFeature:util.ArrayList[Array[Feature]])
    }
    result
  }


  def main(args: Array[String]): Unit = {

    startClient()

    val input_user = fromFile(args(0))
    val input_app = fromFile(args(1))

    val appfea = mutable.ArrayBuilder.make[(String,String)]
    val userfea = mutable.ArrayBuilder.make[(String,java.util.Map[String,String])]

    for (line <- input_app.getLines()) {
      val tmp = (line.split(",").head,line)
      appfea += tmp
    }

    val appf = appfea.result()
    println(appf.toList)
    val appids: Array[String] = appf.map(_._1).take(args(2).toInt)

    //val appids: Array[String] = appf.map(_._1)

    val appf0 = AppFeatureToFeatures.changeAppFeatureToFeature(appf.toMap.asJava)

    for (line <- input_user.getLines()){
      val line1 = if (line.endsWith(",")){
        line+"0"
      }else{
        line
      }
      val tmp = line1.toString.split(",")
      val install = tmp(1)
      val uninstall = tmp(2)
      val phonetype = tmp(3)
      val search = tmp(4)
      val appstore_search = tmp(5)
      val citys = tmp(6)
      val appuse_length = tmp(7)

      val map = new java.util.HashMap[String,String]()
      if(!install.equals("") && install!=null){map.put("install",install)}
      if(!uninstall.equals("") && uninstall!=null){map.put("uninstall",uninstall)}
      if(!phonetype.equals("") && phonetype!=null){map.put("phonetype",phonetype)}
      if(!search.equals("") && search!=null){map.put("search",search)}
      if(!appstore_search.equals("") && appstore_search!=null){map.put("appstore_search",appstore_search)}
      if(!citys.equals("") && citys!=null){map.put("citys",citys)}
      if(!appuse_length.equals("") && appuse_length!=null){map.put("appuse_length",appuse_length)}
      userfea += ((tmp(0),map))
      println(userfea)
      }

    val usef: Array[(String, util.Map[String, String])] = userfea.result()

    val query: String = "快手"
    //创建线程池
    val threadPool:ExecutorService=Executors.newFixedThreadPool(args(3).toInt)

    val start_time = Calendar.getInstance.getTimeInMillis

    for (i <- 0 until math.min(usef.length,args(4).toInt)) {
      val imeifea: String = usef(i)._1
      val userFeature: java.util.Map[String, String] = usef(i)._2
      new ThreadDemo(query, imeifea, userFeature, appids, appf0).run()
    }

    val elapsed_time = Calendar.getInstance.getTimeInMillis - start_time

    println("每个请求平均耗时:"+(elapsed_time / math.min(usef.length,args(4).toInt)))

    stopClient()
    System.exit(0)
  }

  class ThreadDemo(query:String, imeifea:String, userFeature:java.util.Map[String,String],
                   appids: Array[String],appf0: util.Map[String, Array[Feature]]) extends Runnable{
    override def run(){

      val start_time = Calendar.getInstance.getTimeInMillis

      val appFeature = new util.ArrayList[Array[Feature]]()
      appids.foreach(t=>{
        appFeature.add(appf0.get(t))
      })

      val userFeature = new util.HashMap[String,String]()

      val result = computeScores(query, imeifea, userFeature , appFeature)
      val elapsed_time = Calendar.getInstance.getTimeInMillis - start_time
      println(result)
//      if(elapsed_time>120){
//        println("超时："+elapsed_time+"ms")
//      }
    }
  }
}



