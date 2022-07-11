import org.apache.spark._
import org.apache.spark.SparkConf
import scala.math
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.{RowMatrix,IndexedRowMatrix, MatrixEntry, CoordinateMatrix}
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import java.io.FileWriter
import scala.util.Marshal
import scala.io.Source
import scala.collection.immutable
import java.io._
import com.github.fommil.netlib.BLAS
import org.netlib.util.{doubleW, intW}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.github.fommil.netlib.ARPACK


object KirchhoffApp {
def main(args: Array[String]) {
	val conf = new SparkConf().setAppName("KirchhoffApp")
	val sc = new SparkContext(conf) 

	class ParsePDB(val PDBname: String){
		/** PARSE PDB Class */
		var textFile = sc.textFile(PDBname);
		var pairs=textFile.map(line => (line.split("\\s+")(0),line));
		//var ATOMlines = pairs.filter{case (key, value) => value.split("\\s+")(0)=="ATOM"};
                var CAATOMlines = pairs.filter{case (key, value) => value.split("\\s+")(0)=="ATOM" && value.split("\\s+")(2)=="CA"};
		def coors = CAATOMlines.map{case (key, value) => (value.toString.substring(30,38).toFloat, value.toString.substring(39,46).toFloat,value.toString.substring(47,54).toFloat )};
		}

	var t0 = System.nanoTime()
	/** Parse PDB */	
	var PDBfile= new ParsePDB(args(0))
	var coors = PDBfile.coors.cache();

     
	var cutoff=args(1).toInt;
	var gamma=args(2);
	val mb = 1024*1024;
	
	/** Define Null Coordinate Matrix*/
	var coo_matrix_input:org.apache.spark.rdd.RDD[(Int, Int, Float)] = sc.parallelize(Array[(Int, Int, Float)]());

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Build Hessian
	
	
	var Coors_ind = coors.zipWithIndex.map {case (values, key) => (key.toInt, values)};
	var Combs=Coors_ind.cartesian(Coors_ind).filter{ case (x,y) => x._1 < y._1 };
	var joinReady_Combs = Combs.map{case (x,y) => ((x._1, y._1), (x._2,y._2))};

	/** Compute RMSD*/							
	var rdd_cartesian_O=Combs.map{ case ((id1,(x1, y1,z1)), (id2,(x2, y2,z2))) => ((id1, id2), (x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2))};
	var rdd2=rdd_cartesian_O.filter{case(x,y) => y < cutoff*cutoff};
	var rdd3= rdd2.filter{case(x,y) => y!= 0};

	var Combs_rel = joinReady_Combs.join(rdd3).mapValues(_._1);
	var Combs_rmsd = Combs_rel.join(rdd3);
	Combs_rmsd.cache();
	var rdd_cartesian = Combs_rmsd.map{ case (x,y) => (3*x._1+0, 3*x._2+0,(y._1._2._1-y._1._1._1)*(y._1._2._1-y._1._1._1)/(y._2))}.union(Combs_rmsd.map{ case (x,y) => (3*x._1+0, 3*x._2+1,(y._1._2._1-y._1._1._1)*(y._1._2._2-y._1._1._2)/(y._2))}).union(Combs_rmsd.map{ case (x,y) => (3*x._1+0, 3*x._2+2,(y._1._2._1-y._1._1._1)*(y._1._2._3-y._1._1._3)/(y._2))}).union(Combs_rmsd.map{ case (x,y) => (3*x._1+1, 3*x._2+0,(y._1._2._2-y._1._1._2)*(y._1._2._1-y._1._1._1)/(y._2))}).union(Combs_rmsd.map{ case (x,y) => (3*x._1+1, 3*x._2+1,(y._1._2._2-y._1._1._2)*(y._1._2._2-y._1._1._2)/(y._2))}).union(Combs_rmsd.map{ case (x,y) => (3*x._1+1, 3*x._2+2,(y._1._2._2-y._1._1._2)*(y._1._2._3-y._1._1._3)/(y._2))}).union(Combs_rmsd.map{ case (x,y) => (3*x._1+2, 3*x._2+0,(y._1._2._3-y._1._1._3)*(y._1._2._1-y._1._1._1)/(y._2))}).union(Combs_rmsd.map{ case (x,y) => (3*x._1+2, 3*x._2+1,(y._1._2._3-y._1._1._3)*(y._1._2._2-y._1._1._2)/(y._2))}).union(Combs_rmsd.map{ case (x,y) => (3*x._1+2, 3*x._2+2,(y._1._2._3-y._1._1._3)*(y._1._2._3-y._1._1._3)/(y._2))});

	coo_matrix_input = coo_matrix_input.union(rdd_cartesian);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Transpose the matrix

	var coo_matrix_input_LT = coo_matrix_input.map{ case (i,j,k) => (j,i,k)};
	var coo_matrix_input_all = coo_matrix_input_LT.union(coo_matrix_input);
	coo_matrix_input_all.cache()

	// Diagonalize RDD

	var diag_entries_1 = coo_matrix_input_all.filter{case (row, col, value) => col%3 ==0}.map{case (row, _, value) => (row, value)}.reduceByKey(_ + _).map{case (row,value) => (row, 3*(row/3),-value )};
	diag_entries_1.cache()
	var diag_entries_2 = coo_matrix_input_all.filter{case (row, col, value) => col%3 ==1}.map{case (row, _, value) => (row, value)}.reduceByKey(_ + _).map{case (row,value) => (row, 3*(row/3)+1,-value )};
	diag_entries_2.cache()
	var diag_entries_3 = coo_matrix_input_all.subtract(diag_entries_1.union(diag_entries_2)).map{case (row, _, value) => (row, value)}.reduceByKey(_ + _).map{case (row,value) => (row, 3*(row/3)+2,-value )};

	//Diagonalize RDD
	var diag_entries = diag_entries_1.union(diag_entries_2).union(diag_entries_3);

	coo_matrix_input_all  = coo_matrix_input_all.union(diag_entries);
	var coo_matrix_entries = coo_matrix_input_all.map(e => MatrixEntry(e._1, e._2, e._3));
	var coo_matrix = new CoordinateMatrix(coo_matrix_entries);

	//SAVE TO A FILE
        coo_matrix.entries.repartition(1).saveAsTextFile("./Hessian_V11_4v7o_4cores_1");

	var runtime = Runtime.getRuntime;
        var t2 = System.nanoTime()
        println("Elapsed time for construction: " + (t2 - t0)/1000000000.0 + "s")

	val dataRows = coo_matrix.toRowMatrix.rows

	//Singular value decomposition
        var k = args(3).toInt; //N_singvalues

        val svd = new RowMatrix(dataRows.persist()).computeSVD(k, computeU = true)
        val U: RowMatrix = svd.U // The U factor is a RowMatrix.
        val s: Vector = svd.s // The singular values are stored in a local dense vector.
        val V: Matrix = svd.V //The V factor is a local dense matrix.

        // //Save to a file
        val s1=s.toArray;
        val s2= sc.parallelize(s1);
        s2.repartition(1).saveAsTextFile("EigenValues_4v7o_4cores");
        val v1=V.toArray;
        val v2= sc.parallelize(v1);
        v2.repartition(1).saveAsTextFile("EigenVectors_4v7o_4cores");

	runtime = Runtime.getRuntime;
        var t4 = System.nanoTime()
        println("Elapsed time for SVD: " + (t4 - t2)/1000000000.0 + "s")
	System.out.println("New session1,total memory = %s, used memory = %s, free memory = %s".format(runtime.totalMemory/mb, (runtime.totalMemory - runtime.freeMemory) / mb, runtime.freeMemory/mb));
}
}

