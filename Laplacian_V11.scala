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
		def chain_coor = CAATOMlines.map{case (key, value) => (value.toString.substring(20,21),value.toString.substring(30,38).toFloat, value.toString.substring(39,46).toFloat,value.toString.substring(47,54).toFloat )};
		def Chains_Natoms=CAATOMlines.map{case (key, value) => (value.toString.substring(20,21),1)}.reduceByKey(_ + _).sortByKey();
		def Chains = Chains_Natoms.map(line => line._1);
		def Natoms = Chains_Natoms.map(line => line._2);
		}

	var t0 = System.nanoTime()
	/** Parse PDB */	
	var PDBfile= new ParsePDB(args(0))
	var chain_coor = PDBfile.chain_coor.cache();
	var Chains_Natoms= PDBfile.Chains_Natoms;
	var Chains = PDBfile.Chains;
	var Natoms = PDBfile.Natoms;

	/** Get Relevant Indices For Final Matrix*/
	var Mat_indices:Array[Int]=Array();
	for(i <- 0 until (Natoms.collect().length +1))
   	{Mat_indices = Mat_indices:+ Natoms.collect().take(i).sum ; }
     
	var Total_atoms=0;
	var cutoff=args(1).toInt;
	var gamma=args(2);
	val mb = 1024*1024;
	
	/** Define Null Coordinate Matrix*/
    var coo_matrix_input:org.apache.spark.rdd.RDD[(Int, Int, Float)] = sc.parallelize(Array[(Int, Int, Float)]());

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Build Hessian
	
	for ((chain,count1) <- Chains.collect().zipWithIndex) {
		var nrow=Natoms.collect()(count1);
		Total_atoms += nrow;

		/** Get All Chain Combinations*/
		var combinations = Chains.cartesian(Chains).filter{ case (x,y) => x==chain & x<=y };

		for ((comb,count2) <- combinations.collect().zipWithIndex) {
			println(comb);
			//Build Diagonal Hessian
			if (comb._1==comb._2) {
				var coord=chain_coor.filter(chain_coor=> chain_coor._1==comb._1).map(x=> (x._2,x._3,x._4));		
				var Coord_ind = coord.zipWithIndex.map{case (values, key) => (values, key.toInt)};
				var Combs=Coord_ind.cartesian(Coord_ind);
				
				/** Compute RMSD*/
				var rdd_cartesian=Combs.map{ case (((x1, y1,z1),id1), ((x2, y2,z2),id2)) => (id1, id2, math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2)).toFloat)};
				var rdd2=rdd_cartesian.map{x => if (x._3 < cutoff) (x._1, x._2,-gamma.toFloat) else (x._1, x._2,0.toFloat)};
				var rdd3= rdd2.filter{x => (x._3!= 0)};
				
				/** Get Coordinate Matrix Format*/
				coo_matrix_input = coo_matrix_input.union(rdd3.map{case(i,j,v)=> (i + Mat_indices(count1),j+Mat_indices(count2 + count1),v)});
				coo_matrix_input.cache();
				} 

			else {
				/** Build Non-Diagonal Hessian */
				var coord1=chain_coor.filter(chain_coor=> chain_coor._1==comb._1).map(x=> (x._2,x._3,x._4)).zipWithIndex.map {case (values, key) => (values, key.toInt)};
				var coord2=chain_coor.filter(chain_coor=> chain_coor._1==comb._2).map(x=> (x._2,x._3,x._4)).zipWithIndex.map {case (values, key) => (values, key.toInt)};
				var Combs=coord1.cartesian(coord2);

				var rdd_cartesian=Combs.map{ case (((x1, y1,z1),id1), ((x2, y2,z2),id2)) => (id1, id2, math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2)).toFloat)};
				var rdd2=rdd_cartesian.map{x => if (x._3 < cutoff) (x._1, x._2,-gamma.toFloat) else (x._1, x._2,0.toFloat)};
				var rdd3= rdd2.filter{x => (x._3!= 0)};
				
				/** Get Coordinate Matrix Format*/
				coo_matrix_input = coo_matrix_input.union(rdd3.map{case(i,j,v)=> (i + Mat_indices(count1),j+Mat_indices(count2 + count1),v)});
				coo_matrix_input.cache();
				} 
			}
		}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Transpose the matrix
    var coo_matrix_input_LT = coo_matrix_input.map{ case (i,j,k) => (j,i,k)};
    var coo_matrix_input_all = coo_matrix_input_LT.union(coo_matrix_input).distinct();
    //coo_matrix_input_all.cache()
   
    // Diagonalize RDD  
    var diag_entries = coo_matrix_input_all.map{case (row, _, value) => (row, value)}.reduceByKey(_ + _).map{case (row,value) => (row, row,-value -1)};
    var nondiag_entries = coo_matrix_input_all.filter{case (i,j,k) => i!=j};

    coo_matrix_input_all  = nondiag_entries.union(diag_entries);

    //SAVE TO A FILE
    coo_matrix_input_all.repartition(1).saveAsTextFile("./Laplacian_V9_4v7o_16cores_1")
	var runtime = Runtime.getRuntime;
    var t2 = System.nanoTime()
    println("Elapsed time for construction: " + (t2 - t0)/1000000000.0 + "s")

	
	// //Singular value decomposition
	
	var coo_matrix_entries = coo_matrix_input_all.map(e => MatrixEntry(e._1, e._2, e._3));
    var coo_matrix = new CoordinateMatrix(coo_matrix_entries);
	val dataRows = coo_matrix.toRowMatrix.rows

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
    println("System size: " + Natoms.sum() + "atoms")
    println("No. of chains: " + Chains.count())


}
}
