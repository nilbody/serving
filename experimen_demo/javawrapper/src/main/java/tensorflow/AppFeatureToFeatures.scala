package tensorflow

import java.util

import com.google.protobuf.ByteString
import org.tensorflow.example.Feature

import scala.collection.{JavaConversions, mutable}
import scala.collection.JavaConverters._

object AppFeatureToFeatures {

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

  /**
    *
    * @param map<String,String>
    * @return map<String,Array<Feature>>
    */
  def changeAppFeatureToFeature(map:java.util.Map[String,String]): java.util.Map[String, Array[Feature]] ={
    val input = JavaConversions.mapAsScalaMap(map)
    val out = new java.util.HashMap[String, Array[Feature]]()
    input.foreach(t =>{
      val app0 = if (t._2.endsWith(",")) t._2 + "0" else t._2
      val app = app0.split(",")
      if (app.length == 3){
        //v1,v2版本，只有3个字段
        val appid = createIntFeature(if (!app(0).equals("") && app(0) != null) app(0).split(";").map(_.toInt) else Array[Int](0))
        val appname_indices = createByteStringFeature(app(1).split(";").map(t => ByteString.copyFrom(t.getBytes)))
        val appfeature = createIntFeature(if (!app(2).equals("") && app(2) != null) app(2).split(";").map(_.toInt) else Array[Int](0))
        val feature: Array[Feature] = Array(appid, appname_indices, appfeature)
        out.put(t._1,feature)
      }else if(app.length == 4){
        //v3版本，有4个字段，保存了相似度的版本
        val appid = createIntFeature(if (!app(0).equals("") && app(0) != null) app(0).split(";").map(_.toInt) else Array[Int](0))
        val appname_indices = createByteStringFeature(Array(app(1)).map(t => ByteString.copyFrom(t.getBytes)))
        val appfeature = createIntFeature(if (!app(2).equals("") && app(2) != null) app(2).split(";").map(_.toInt) else Array[Int](0))
        val simility = createFloatFeature(if (!app(3).equals("") && app(3) != null) app(3).split(";").map(_.toFloat) else Array[Float](0))
        val feature: Array[Feature] = Array(appid, appname_indices, appfeature, simility)
        out.put(t._1,feature)
      }
    })
    out
  }

}
